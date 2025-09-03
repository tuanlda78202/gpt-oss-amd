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

#include "../include/pp_manager.hpp"

static float* forward_hybrid_stage(OssTransformerHybrid* shard, int global_l_start, bool has_embed,
                                   bool has_out, int token, int pos);

// Single hidden vector hop between devices
static void pp_pass_x_between(PPManager* mgr, int from_idx, int to_idx) {
    auto& A = mgr->shards[from_idx];
    auto& B = mgr->shards[to_idx];
    size_t bytes = (size_t)mgr->hidden_dim * sizeof(float);
    if (mgr->peer_ok[A.device_id][B.device_id]) {
        CHECK_HIP(hipMemcpyPeerAsync(B.shard->state.x, B.device_id, A.shard->state.x, A.device_id,
                                     bytes, 0));
        CHECK_HIP(hipDeviceSynchronize());
    } else {
        if (!mgr->host_bounce)
            CHECK_HIP(hipHostMalloc((void**)&mgr->host_bounce, bytes));
        CHECK_HIP(hipSetDevice(A.device_id));
        CHECK_HIP(hipMemcpy(mgr->host_bounce, A.shard->state.x, bytes, hipMemcpyDeviceToHost));
        CHECK_HIP(hipSetDevice(B.device_id));
        CHECK_HIP(hipMemcpy(B.shard->state.x, mgr->host_bounce, bytes, hipMemcpyHostToDevice));
    }
}

static float* forward_token_pipeline(OssTransformerHybrid* root, int token, int pos) {
    PPManager* mgr = root->pp;
    // Stage 0
    CHECK_HIP(hipSetDevice(mgr->shards[0].device_id));
    float* ret = forward_hybrid_stage(mgr->shards[0].shard, mgr->shards[0].l_start,
                                      mgr->shards[0].has_embed, mgr->shards[0].has_out, token, pos);
    // Middle stages
    for (int p = 1; p < mgr->pp; ++p) {
        pp_pass_x_between(mgr, p - 1, p);
        CHECK_HIP(hipSetDevice(mgr->shards[p].device_id));
        ret = forward_hybrid_stage(mgr->shards[p].shard, mgr->shards[p].l_start,
                                   mgr->shards[p].has_embed, mgr->shards[p].has_out, token, pos);
    }
    return ret; // logits on last stage
}

// ---------- your existing kernels are reused inside this ----------
static float* forward_hybrid_stage(OssTransformerHybrid* transformer, int global_l_start,
                                   bool has_embed, bool has_out, int token, int pos) {
    OssConfig* p = &transformer->config;
    OssTransformerWeightsHalf* w = &transformer->weights;
    OssRunState* s = &transformer->state;

    float* x = s->x;
    int head_dim = p->head_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int n_local_layers = p->n_layers;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;

    float *cos_vals = nullptr, *sin_vals = nullptr;
    CHECK_HIP(hipMalloc(&cos_vals, (size_t)(head_dim / 2) * sizeof(float)));
    CHECK_HIP(hipMalloc(&sin_vals, (size_t)(head_dim / 2) * sizeof(float)));

    if (has_embed) {
        // your existing embed path (FP16->FP32 add)
        embed_gpu(x, w->token_embedding_table + (size_t)token * hidden_dim, hidden_dim);
    }

    for (int ll = 0; ll < n_local_layers; ++ll) {
        int l_global = global_l_start + ll;

        rmsnorm_gpu(s->t, x, w->rms_attn_w + (size_t)ll * hidden_dim, hidden_dim);

        size_t loff_local = (size_t)ll * p->seq_len * kv_dim;
        s->k = s->key_cache + loff_local + (size_t)pos * kv_dim;
        s->v = s->value_cache + loff_local + (size_t)pos * kv_dim;

        __half* w_qkv = w->w_qkv + (size_t)ll * hidden_dim *
                                       (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        __half* b_qkv =
            w->b_qkv + (size_t)ll * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        matvec_gpu(s->qkv, s->t, w_qkv, b_qkv, hidden_dim,
                   (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

        split_qkv_gpu(s->qkv, s->q, s->k, s->v, head_dim, p->n_attn_heads, p->n_kv_heads);

        float ntk_beta = 32.0f, ntk_alpha = 1.0f;
        compute_cosin_gpu(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                          p->initial_context_length, ntk_beta, ntk_alpha, cos_vals, sin_vals);
        rope_gpu(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
        rope_gpu(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);

        flash_attn_decode_gpu(s->q, s->key_cache + (size_t)ll * p->seq_len * kv_dim,
                              s->value_cache + (size_t)ll * p->seq_len * kv_dim, s->mask,
                              w->attn_sinks + (size_t)ll * p->n_attn_heads, s->tb, pos, p->seq_len,
                              head_dim, kv_dim, kv_mul, p->sliding_window, l_global,
                              p->n_attn_heads);

        __half* w_o = w->w_o + (size_t)ll * (head_dim * p->n_attn_heads) * hidden_dim;
        __half* b_o = w->b_o + (size_t)ll * hidden_dim;
        matvec_gpu(s->tb2, s->tb, w_o, b_o, head_dim * p->n_attn_heads, hidden_dim);
        vec_add_vec_gpu(x, s->tb2, 1.0f, hidden_dim);

        // FFN (MoE)
        rmsnorm_gpu(s->t, x, w->rms_ffn_w + (size_t)ll * hidden_dim, hidden_dim);

        __half* w_router = w->w_router + (size_t)ll * hidden_dim * p->n_experts;
        __half* b_router = w->b_router + (size_t)ll * p->n_experts;
        matvec_gpu(s->router_score, s->t, w_router, b_router, hidden_dim, p->n_experts);

        topk_gpu(s->topk_v, s->topk_i, s->router_score, p->n_experts, p->experts_per_token);
        softmax_gpu(s->topk_v, p->experts_per_token);

        CHECK_HIP(hipMemset(s->e_agg, 0, (size_t)hidden_dim * sizeof(float)));
        int safe_k = p->experts_per_token > 16 ? 16 : p->experts_per_token;
        int topk_idx[16];
        float topk_w[16];
        CHECK_HIP(hipMemcpy(topk_idx, s->topk_i, safe_k * sizeof(int), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(topk_w, s->topk_v, safe_k * sizeof(float), hipMemcpyDeviceToHost));

        for (int t = 0; t < safe_k; ++t) {
            int e = topk_idx[t];
            float ew = topk_w[t];
            __half* w_mlp1 = w->w_mlp1 + (size_t)(ll * p->n_experts + e) *
                                             (2 * p->intermediate_dim) * hidden_dim;
            __half* b_mlp1 =
                w->b_mlp1 + (size_t)(ll * p->n_experts + e) * (2 * p->intermediate_dim);
            matvec_gpu(s->mlp1_out, s->t, w_mlp1, b_mlp1, hidden_dim, 2 * p->intermediate_dim);

            split_gate_up_gpu(s->mlp1_out, s->gate, s->up, p->intermediate_dim);
            const float alpha = 1.702f;
            swiglu_gpu(s->gate, s->up, p->intermediate_dim, alpha, p->swiglu_limit);
            CHECK_HIP(hipMemcpy(s->gate_up, s->gate, (size_t)p->intermediate_dim * sizeof(float),
                                hipMemcpyDeviceToDevice));

            __half* w_mlp2 =
                w->w_mlp2 + (size_t)(ll * p->n_experts + e) * hidden_dim * p->intermediate_dim;
            __half* b_mlp2 = w->b_mlp2 + (size_t)(ll * p->n_experts + e) * hidden_dim;
            matvec_gpu(s->tb2, s->gate_up, w_mlp2, b_mlp2, hidden_dim, p->intermediate_dim);

            vec_add_vec_gpu(s->e_agg, s->tb2, ew, hidden_dim);
        }
        vec_add_vec_gpu(x, s->e_agg, 1.0f, hidden_dim);
    }

    if (has_out) {
        rmsnorm_gpu(x, x, w->rms_out_w, hidden_dim);
        matvec_gpu(s->logits, x, w->out, nullptr, hidden_dim, p->vocab_size);
    }

    if (cos_vals)
        CHECK_HIP(hipFree(cos_vals));
    if (sin_vals)
        CHECK_HIP(hipFree(sin_vals));
    return has_out ? s->logits : s->x;
}

// ---------- transparent top-level forward ----------
float* forward_hybrid(OssTransformerHybrid* transformer, int token, int pos) {
    if (transformer->pp != nullptr) {
        return forward_token_pipeline(transformer, token, pos);
    }
    // else: your original single-GPU forward_hybrid body here (unchanged)
    // ...
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

        // ! MHA (legacy -> FA)
        // compute_attn_gpu(s->q, s->key_cache + loff, s->att, s->mask, pos, p->seq_len, head_dim,
        // kv_dim, kv_mul, p->sliding_window, l, p->n_attn_heads);

        // add_attn_sink_gpu(s->att, w->attn_sinks + l * p->n_attn_heads, pos, p->seq_len,
        // p->n_attn_heads, l);

        // softmax_attn_gpu(s->att, pos, p->seq_len, p->n_attn_heads);

        // w_value_acc_gpu(s->att, s->value_cache + loff, s->tb, pos, p->seq_len, head_dim, kv_dim,
        // kv_mul, p->n_attn_heads);

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
