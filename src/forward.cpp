#include "../include/model.hpp"
#include "hip/BLAS.hip"
#include "hip/attention.hip"
#include "hip/embed.hip"
#include "hip/matvec.hip"
#include "hip/prim_add.hip"
#include "hip/rmsnorm.hip"
#include "hip/rope.hip"
#include "hip/softmax.hip"
#include "hip/swilglu.hip"
#include "hip/topk.hip"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

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
        rmsnorm_hip_hybrid_device(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

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

        matmul_hip_hybrid_device(s->qkv, s->t, w_qkv, b_qkv, hidden_dim,
                                 (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

        // ! Separate q, k, v on GPU
        CHECK_HIP(hipMemcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float),
                            hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(s->k, s->qkv + head_dim * p->n_attn_heads,
                            head_dim * p->n_kv_heads * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads,
                            head_dim * p->n_kv_heads * sizeof(float), hipMemcpyDeviceToDevice));

        // ! RoPE
        float ntk_beta = 32.0f;
        float ntk_alpha = 1.0f;

        compute_cos_sin_hip_device(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                                   p->initial_context_length, ntk_beta, ntk_alpha, cos_vals,
                                   sin_vals);

        apply_rotary_emb_hip_device(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
        apply_rotary_emb_hip_device(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);

        // ! MHA
        compute_attention_scores_hip_device(s->q, s->key_cache + loff, s->att, s->mask, pos,
                                            p->seq_len, head_dim, kv_dim, kv_mul, p->sliding_window,
                                            l, p->n_attn_heads);

        add_attention_sink_hip_device(s->att, w->attn_sinks + l * p->n_attn_heads, pos, p->seq_len,
                                      p->n_attn_heads, l);

        softmax_attention_hip_device(s->att, pos, p->seq_len, p->n_attn_heads);

        weighted_value_accumulation_hip_device(s->att, s->value_cache + loff, s->tb, pos,
                                               p->seq_len, head_dim, kv_dim, kv_mul,
                                               p->n_attn_heads);

        // ! Output projection
        __half* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        __half* b_o = w->b_o + 1ll * l * hidden_dim;

        matmul_hip_hybrid_device(s->tb2, s->tb, w_o, b_o, head_dim * p->n_attn_heads, hidden_dim);

        // ! Residual connection
        vecaddvec_hip_device(x, s->tb2, 1.0f, hidden_dim);

        // ! RMSNorm
        rmsnorm_hip_hybrid_device(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

        // ! MoE Router
        __half* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        __half* b_router = w->b_router + 1ll * l * n_experts;

        matmul_hip_hybrid_device(s->router_score, s->t, w_router, b_router, hidden_dim, n_experts);

        topk_hip_device(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);
        softmax_hip_device(s->topk_v, p->experts_per_token);

        // ! Expert processing
        CHECK_HIP(hipMemset(s->e_agg, 0, hidden_dim * sizeof(float)));

        // TODO (explain): Copy topk_indices to host once per layer
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

            matmul_hip_hybrid_device(s->mlp1_out, s->t, w_mlp1, b_mlp1, hidden_dim,
                                     2 * p->intermediate_dim);

            // Split mlp1_out into gate and up
            split_gate_up_hip_device(s->mlp1_out, s->gate, s->up, p->intermediate_dim);

            // SwiGLU non-linearity
            const float alpha = 1.702f;
            thaDNN_s_swiglu(s->gate, s->up, p->intermediate_dim, alpha, p->swiglu_limit);

            // Copy result back to gate_up buffer
            CHECK_HIP(hipMemcpy(s->gate_up, s->gate, p->intermediate_dim * sizeof(float),
                                hipMemcpyDeviceToDevice));

            // ! Final matmul (down project)
            long long w_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim * p->intermediate_dim;
            long long b_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim;

            __half* w_mlp2 = w->w_mlp2 + w_mlp2_offset;
            __half* b_mlp2 = w->b_mlp2 + b_mlp2_offset;

            matmul_hip_hybrid_device(s->tb2, s->gate_up, w_mlp2, b_mlp2, hidden_dim,
                                     p->intermediate_dim);

            // Aggregate expert
            vecaddvec_hip_device(s->e_agg, s->tb2, expert_w, hidden_dim);
        }

        // Residual connection
        vecaddvec_hip_device(x, s->e_agg, 1.0f, hidden_dim);
    }

    // Final operations
    rmsnorm_hip_hybrid_device(x, x, w->rms_out_w, hidden_dim);

    // Linear: classifier into logits
    matmul_hip_hybrid_device(s->logits, x, w->out, nullptr, hidden_dim, p->vocab_size);

    CHECK_HIP(hipDeviceSynchronize());

    if (cos_vals) {
        CHECK_HIP(hipFree(cos_vals));
    }
    if (sin_vals) {
        CHECK_HIP(hipFree(sin_vals));
    }

    return s->logits;
}
