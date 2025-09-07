#include "../include/model.hpp"
#include "hip/BLAS.hip"
#include "hip/attention.hip"
#include "hip/embed.hip"
#include "hip/matvec.hip"
#include "hip/prim_add.hip"
#include "hip/rmsnorm.hip"
#include "hip/rope.hip"
#include "hip/softmax.hip"
#include "hip/split_qkv.hip"
#include "hip/swilglu.hip"
#include "hip/topk.hip"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

// TODO: merge interface batch=1
float* forward_hybrid(OssTransformerHybrid* transformer, int token, int pos) {
    OssConfig* p = &transformer->config;
    OssTransformerWeightsHalf* w = &transformer->weights;
    OssRunState* s = &transformer->state;

    float* x = s->x;
    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;

    float* cos_vals = nullptr;
    float* sin_vals = nullptr;
    size_t cos_sin_size = (head_dim / 2) * sizeof(float);

    CHECK_HIP(hipMalloc(&cos_vals, cos_sin_size));
    CHECK_HIP(hipMalloc(&sin_vals, cos_sin_size));

    // ! Embedding
    embed_gpu(x, w->token_embedding_table + token * hidden_dim, hidden_dim);

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // ! RMSNorm
        rmsnorm_gpu(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

        // ! KV cache management
        int loff = l * p->seq_len * kv_dim;
        long long kv_offset = loff + pos * kv_dim;
        s->k = s->key_cache + kv_offset;
        s->v = s->value_cache + kv_offset;

        // ! QKV project
        __half* w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                       (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        __half* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

        matvec_gpu(s->qkv, s->t, w_qkv, b_qkv, hidden_dim,
                   (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

        // ! Split QKV
        split_qkv_gpu(s->qkv, s->q, s->k, s->v, head_dim, p->n_attn_heads, p->n_kv_heads);

        // ! RoPE
        float ntk_beta = 32.0f;
        float ntk_alpha = 1.0f;

        compute_cosin_gpu(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                          p->initial_context_length, ntk_beta, ntk_alpha, cos_vals, sin_vals);
        rope_gpu(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
        rope_gpu(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);

        // ! GQA
        flash_attn_decode_gpu(s->q, s->key_cache + loff, s->value_cache + loff, s->mask,
                              w->attn_sinks + l * p->n_attn_heads, s->tb, pos, p->seq_len, head_dim,
                              kv_dim, kv_mul, p->sliding_window, l, p->n_attn_heads);

        // ! Output projection
        __half* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        __half* b_o = w->b_o + 1ll * l * hidden_dim;

        matvec_gpu(s->tb2, s->tb, w_o, b_o, head_dim * p->n_attn_heads, hidden_dim);

        // ! Residual connection
        vec_add_vec_gpu(x, s->tb2, 1.0f, hidden_dim);

        // ! RMSNorm
        rmsnorm_gpu(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

        // ! MoE Router
        __half* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        __half* b_router = w->b_router + 1ll * l * n_experts;

        matvec_gpu(s->router_score, s->t, w_router, b_router, hidden_dim, n_experts);

        topk_gpu(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);
        softmax_gpu(s->topk_v, p->experts_per_token);

        // ! Expert processing
        CHECK_HIP(hipMemset(s->e_agg, 0, hidden_dim * sizeof(float)));

        int topk_indices_host[16];
        float topk_weights_host[16];
        int safe_experts_per_token = (p->experts_per_token > 16) ? 16 : p->experts_per_token;

        CHECK_HIP(hipMemcpy(topk_indices_host, s->topk_i, safe_experts_per_token * sizeof(int),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(topk_weights_host, s->topk_v, safe_experts_per_token * sizeof(float),
                            hipMemcpyDeviceToHost));

        for (int idx = 0; idx < safe_experts_per_token; idx++) {
            int e = topk_indices_host[idx];
            float expert_w = topk_weights_host[idx];

            // ! Linear 1 (Gated MLP)
            long long w_mlp1_offset =
                1ll * (l * n_experts + e) * (2 * p->intermediate_dim) * hidden_dim;
            long long b_mlp1_offset = 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);

            __half* w_mlp1 = w->w_mlp1 + w_mlp1_offset;
            __half* b_mlp1 = w->b_mlp1 + b_mlp1_offset;

            matvec_gpu(s->mlp1_out, s->t, w_mlp1, b_mlp1, hidden_dim, 2 * p->intermediate_dim);

            // ! Split mlp1_out into gate and up
            split_gate_up_gpu(s->mlp1_out, s->gate, s->up, p->intermediate_dim);

            // ! SwiGLU non-linearity
            const float alpha = 1.702f;
            swiglu_gpu(s->gate, s->up, p->intermediate_dim, alpha, p->swiglu_limit);

            // ! Copy result back to gate_up buffer
            CHECK_HIP(hipMemcpy(s->gate_up, s->gate, p->intermediate_dim * sizeof(float),
                                hipMemcpyDeviceToDevice));

            // ! Final matmul (down project)
            long long w_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim * p->intermediate_dim;
            long long b_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim;

            __half* w_mlp2 = w->w_mlp2 + w_mlp2_offset;
            __half* b_mlp2 = w->b_mlp2 + b_mlp2_offset;

            matvec_gpu(s->tb2, s->gate_up, w_mlp2, b_mlp2, hidden_dim, p->intermediate_dim);

            // ! Aggregate expert
            vec_add_vec_gpu(s->e_agg, s->tb2, expert_w, hidden_dim);
        }

        // Residual connection
        vec_add_vec_gpu(x, s->e_agg, 1.0f, hidden_dim);
    }

    // Final operations
    rmsnorm_gpu(x, x, w->rms_out_w, hidden_dim);

    // Linear: classifier into logits
    matvec_gpu(s->logits, x, w->out, nullptr, hidden_dim, p->vocab_size);

    if (cos_vals) {
        CHECK_HIP(hipFree(cos_vals));
    }
    if (sin_vals) {
        CHECK_HIP(hipFree(sin_vals));
    }

    return s->logits;
}

float* forward_hybrid_batch(OssTransformerHybrid* transformer, int* tokens, int pos, int batch_size,
                            const int* batch_indices_h, int B_stride) {
    OssConfig* p = &transformer->config;
    OssTransformerWeightsHalf* w = &transformer->weights;
    OssRunState* s = &transformer->state;

    float* x = s->x;
    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;

    size_t cos_sin_size = (head_dim / 2) * sizeof(float);

    // ! Copy batch indices to persistent GPU buffer (only when data changes)
    CHECK_HIP(hipMemcpy(s->d_batch_indices, batch_indices_h, batch_size * sizeof(int),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s->d_tokens, tokens, batch_size * sizeof(int), hipMemcpyHostToDevice));

    // ! Embedding - batched
    embed_batch_gpu(x, w->token_embedding_table, s->d_tokens, batch_size, hidden_dim);

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // ! RMSNorm - batched
        rmsnorm_batch_gpu(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);

        // ! QKV project - batched
        __half* w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                       (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        __half* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

        matvec_batch_gpu(s->qkv, s->t, w_qkv, b_qkv, batch_size, hidden_dim,
                         (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

        // ! Split QKV and write to KV cache
        split_qkv_batch_gpu(s->qkv, s->q, s->key_cache, s->value_cache, batch_size, head_dim,
                            p->n_attn_heads, p->n_kv_heads, l, pos, p->seq_len, s->d_batch_indices,
                            B_stride);

        // ! RoPE - batched
        float ntk_beta = 32.0f;
        float ntk_alpha = 1.0f;

        compute_cosin_gpu(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                          p->initial_context_length, ntk_beta, ntk_alpha, s->cos_vals, s->sin_vals);
        // TODO: 1 RoPE only
        rope_q_batch_gpu(s->q, s->cos_vals, s->sin_vals, batch_size, p->n_attn_heads, head_dim);
        rope_k_batch_gpu(s->key_cache, s->cos_vals, s->sin_vals, batch_size, p->n_kv_heads,
                         head_dim, p->seq_len, kv_dim, l, pos, s->d_batch_indices, B_stride);

        // ! GQA - batched (using persistent GPU buffer)
        flash_attn_decode_gpu_batch(s->q, s->key_cache, s->value_cache, s->mask, w->attn_sinks,
                                    s->tb, batch_size, pos, p->seq_len, head_dim, kv_dim, kv_mul,
                                    p->sliding_window, l, p->n_attn_heads, s->d_batch_indices,
                                    B_stride);

        // ! Output projection - batched
        __half* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        __half* b_o = w->b_o + 1ll * l * hidden_dim;

        matvec_batch_gpu(s->tb2, s->tb, w_o, b_o, batch_size, head_dim * p->n_attn_heads,
                         hidden_dim);

        // ! Residual connection - batched
        vec_add_vec_batch_gpu(x, s->tb2, 1.0f, batch_size, hidden_dim);

        // ! RMSNorm - batched
        rmsnorm_batch_gpu(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);

        // ! MoE Router - batched
        __half* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        __half* b_router = w->b_router + 1ll * l * n_experts;

        matvec_batch_gpu(s->router_score, s->t, w_router, b_router, batch_size, hidden_dim,
                         n_experts);

        // ! TopK and Softmax - batched
        topk_batch_gpu(s->topk_v, s->topk_i, s->router_score, batch_size, n_experts,
                       p->experts_per_token);

        softmax_batch_gpu(s->topk_v, batch_size, p->experts_per_token);

        // ! Expert processing
        for (int b = 0; b < batch_size; b++) {
            int batch_offset_experts = b * n_experts;
            int batch_offset_topk = b * p->experts_per_token;
            int batch_offset_hidden = b * hidden_dim;
            int batch_offset_inter = b * intermediate_dim;

            // ! Expert processing for this batch element
            CHECK_HIP(hipMemset(s->e_agg + batch_offset_hidden, 0, hidden_dim * sizeof(float)));

            // TODO: full on GPU
            int topk_indices_host[16];
            float topk_weights_host[16];
            int safe_experts_per_token = (p->experts_per_token > 16) ? 16 : p->experts_per_token;

            CHECK_HIP(hipMemcpy(topk_indices_host, s->topk_i + batch_offset_topk,
                                safe_experts_per_token * sizeof(int), hipMemcpyDeviceToHost));
            CHECK_HIP(hipMemcpy(topk_weights_host, s->topk_v + batch_offset_topk,
                                safe_experts_per_token * sizeof(float), hipMemcpyDeviceToHost));

            // TODO: fused 1 kernel GPU
            for (int idx = 0; idx < safe_experts_per_token; idx++) {
                int e = topk_indices_host[idx];
                float expert_w = topk_weights_host[idx];

                // ! Linear 1 (Gated MLP)
                long long w_mlp1_offset =
                    1ll * (l * n_experts + e) * (2 * p->intermediate_dim) * hidden_dim;
                long long b_mlp1_offset = 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);

                __half* w_mlp1 = w->w_mlp1 + w_mlp1_offset;
                __half* b_mlp1 = w->b_mlp1 + b_mlp1_offset;

                matvec_gpu(s->mlp1_out + b * (2 * intermediate_dim), s->t + batch_offset_hidden,
                           w_mlp1, b_mlp1, hidden_dim, 2 * p->intermediate_dim);

                // ! Split mlp1_out into gate and up
                split_gate_up_gpu(s->mlp1_out + b * (2 * intermediate_dim),
                                  s->gate + batch_offset_inter, s->up + batch_offset_inter,
                                  p->intermediate_dim);

                // ! SwiGLU non-linearity
                const float alpha = 1.702f;
                swiglu_gpu(s->gate + batch_offset_inter, s->up + batch_offset_inter,
                           p->intermediate_dim, alpha, p->swiglu_limit);

                // ! Final matmul (down project)
                long long w_mlp2_offset =
                    1ll * (l * n_experts + e) * hidden_dim * p->intermediate_dim;
                long long b_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim;

                __half* w_mlp2 = w->w_mlp2 + w_mlp2_offset;
                __half* b_mlp2 = w->b_mlp2 + b_mlp2_offset;

                matvec_gpu(s->tb2 + batch_offset_hidden, s->gate + batch_offset_inter, w_mlp2,
                           b_mlp2, hidden_dim, p->intermediate_dim);

                // ! Aggregate expert
                vec_add_vec_gpu(s->e_agg + batch_offset_hidden, s->tb2 + batch_offset_hidden,
                                expert_w, hidden_dim);
            }
        }

        // Residual connection - batched
        vec_add_vec_batch_gpu(x, s->e_agg, 1.0f, batch_size, hidden_dim);
    }

    // Final operations - batched
    rmsnorm_batch_gpu(x, x, w->rms_out_w, batch_size, hidden_dim);

    // Linear: classifier into logits - batched
    matvec_batch_gpu(s->logits, x, w->out, nullptr, batch_size, hidden_dim, p->vocab_size);

    return s->logits;
}
