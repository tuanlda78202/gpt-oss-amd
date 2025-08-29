#include "../include/model.hpp"
#include "BLAS.hip"
#include "hip/attention.hip"
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

bool enable_print = false;
class HipTimer {
  private:
    hipEvent_t start_, stop_;
    const char* label_;

  public:
    HipTimer(const char* label = "Unnamed") : label_(label) {
        hipEventCreate(&start_);
        hipEventCreate(&stop_);
        hipEventRecord(start_, 0);
    }

    ~HipTimer() {
        hipEventRecord(stop_, 0);
        hipEventSynchronize(stop_);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, start_, stop_);

        if (enable_print) {
            printf("[Timer] %s: %.3f ms\n", label_, ms);
        }

        hipEventDestroy(start_);
        hipEventDestroy(stop_);
    }
};

// Kernel to convert FP16 embedding to FP32
__global__ void convert_embedding_kernel(__half* content_row_half, float* x, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        x[idx] = __half2float(content_row_half[idx]);
    }
}

// ! FORWARD (hybrid precision: FP16 weights + FP32 activations)
float* forward_hybrid(OssTransformerHybrid* transformer, int token, int pos) {
    OssConfig* p = &transformer->config;
    OssTransformerWeightsHalf* w = &transformer->weights; // FP16 weights
    OssRunState* s = &transformer->state;                 // FP32 activations

    float* x = s->x;
    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads;
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;

    // CRITICAL FIX: Use per-call allocation to prevent race conditions and memory corruption
    // Static buffers were causing memory access faults due to concurrent access
    float* cos_vals = nullptr;
    float* sin_vals = nullptr;

    size_t cos_sin_size = (head_dim / 2) * sizeof(float);

    CHECK_HIP(hipMalloc(&cos_vals, cos_sin_size));
    CHECK_HIP(hipMalloc(&sin_vals, cos_sin_size));

    // Add safety validation
    if (cos_vals == nullptr || sin_vals == nullptr) {
        printf("ERROR: Failed to allocate cos/sin buffers\n");
        return nullptr;
    }

    // ! Token embedding: copy from FP16 embedding table and convert to FP32
    // Add bounds checking for token embedding access
    if (token < 0 || token >= p->vocab_size) {
        printf("ERROR: Invalid token index %d (vocab_size: %d)\n", token, p->vocab_size);
        // Clean up allocated buffers before returning
        if (cos_vals)
            CHECK_HIP(hipFree(cos_vals));
        if (sin_vals)
            CHECK_HIP(hipFree(sin_vals));
        return nullptr;
    }

    __half* content_row_half = w->token_embedding_table + token * hidden_dim;

    // Convert FP16 embedding to FP32 on GPU with bounds checking
    if (content_row_half == nullptr || x == nullptr) {
        printf("ERROR: NULL pointers in token embedding conversion\n");
        // Clean up allocated buffers before returning
        if (cos_vals)
            CHECK_HIP(hipFree(cos_vals));
        if (sin_vals)
            CHECK_HIP(hipFree(sin_vals));
        return nullptr;
    }

    {
        // HipTimer timer("Token Embedding");
        dim3 embed_block(256);
        dim3 embed_grid((hidden_dim + 255) / 256);
        hipLaunchKernelGGL(convert_embedding_kernel, embed_grid, embed_block, 0, 0,
                           content_row_half, x, hidden_dim);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize()); // Ensure embedding conversion completes
    }

    // Forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // printf("layer %llu\n", l);
        // ! RMSNorm: FP16 weights, FP32 activations
        {
            // HipTimer timer("rmsnorm_attn");
            rmsnorm_hip_hybrid_device(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);
            // float* t_host = (float*)malloc(hidden_dim * sizeof(float));
            // CHECK_HIP(hipMemcpy(t_host, s->t, hidden_dim * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("rmsnorm_attn: "); for (int i = 0; i < 20; i++) {
            //     printf("%f ", t_host[i]);
            // }
            // printf("\n");
            // free(t_host);
        }

        // ! KV cache management: key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim;

        // CRITICAL FIX: Add bounds checking for KV cache access
        long long kv_offset = loff + pos * kv_dim;
        long long max_kv_offset = (long long)p->n_layers * p->seq_len * kv_dim;

        if (kv_offset + kv_dim > max_kv_offset) {
            printf("ERROR: KV cache overflow at pos=%d, layer=%d\n", pos, (int)l);
            printf("  kv_offset=%lld, max_offset=%lld, kv_dim=%d\n", kv_offset, max_kv_offset,
                   kv_dim);
            printf("  seq_len=%d, n_layers=%d\n", p->seq_len, p->n_layers);
            // Clean up allocated buffers before returning
            if (cos_vals)
                CHECK_HIP(hipFree(cos_vals));
            if (sin_vals)
                CHECK_HIP(hipFree(sin_vals));
            return nullptr;
        }

        s->k = s->key_cache + kv_offset;
        s->v = s->value_cache + kv_offset;

        // ! QKV project: FP16 weights -> FP32 computation
        __half* w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                       (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        __half* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

        {
            // HipTimer timer("matmul_qkv");
            matmul_hip_hybrid_device(s->qkv, s->t, w_qkv, b_qkv, hidden_dim,
                                     (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
            // float* qkv_host = (float*)malloc((p->n_attn_heads + 2 * p->n_kv_heads) * head_dim *
            // sizeof(float)); CHECK_HIP(hipMemcpy(qkv_host, s->qkv, (p->n_attn_heads + 2 *
            // p->n_kv_heads) * head_dim * sizeof(float), hipMemcpyDeviceToHost));
            // printf("matmul_qkv: ");
            // for (int i = 0; i < 20; i++) {
            //     printf("%f ", qkv_host[i]);
            // }
            // printf("\n");
            // free(qkv_host);
        }

        // Add bias: FP16 -> FP32 conversion
        // vecaddvec_hip_hybrid_device(s->qkv, b_qkv, 1.0f,
        //                             (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

        // Separate q, k, v on GPU - copy directly to KV cache
        CHECK_HIP(hipMemcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float),
                            hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(s->k, s->qkv + head_dim * p->n_attn_heads,
                            head_dim * p->n_kv_heads * sizeof(float), hipMemcpyDeviceToDevice));
        CHECK_HIP(hipMemcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads,
                            head_dim * p->n_kv_heads * sizeof(float), hipMemcpyDeviceToDevice));

        // ! RoPE relative positional encoding (OPTIMIZED)
        float ntk_beta = 32.0f;
        float ntk_alpha = 1.0f;

        // CRITICAL SAFETY: Validate cos/sin buffers before use
        if (cos_vals == nullptr || sin_vals == nullptr) {
            printf("ERROR: cos/sin buffers are NULL at layer %d\n", (int)l);
            // Clean up any allocated buffer before returning
            if (cos_vals)
                CHECK_HIP(hipFree(cos_vals));
            if (sin_vals)
                CHECK_HIP(hipFree(sin_vals));
            return nullptr;
        }

        // Add bounds checking for RoPE buffer access
        // size_t required_cos_sin_size = (head_dim / 2) * sizeof(float);

        // Use pre-allocated buffers (NO MORE MALLOC/FREE!)
        {
            // HipTimer timer("compute_cos_sin");
            compute_cos_sin_hip_device(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                                       p->initial_context_length, ntk_beta, ntk_alpha, cos_vals,
                                       sin_vals);
        }
        {
            // HipTimer timer("apply_rotary_emb");
            apply_rotary_emb_hip_device(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
            apply_rotary_emb_hip_device(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);
            float* q_host = (float*)malloc(head_dim * p->n_attn_heads * sizeof(float));
            CHECK_HIP(hipMemcpy(q_host, s->q, head_dim * p->n_attn_heads * sizeof(float),
                                hipMemcpyDeviceToHost));
            // printf("apply_rotary_emb_q: ");
            // for (int i = 0; i < 20; i++) {
            //     printf("%f ", q_host[i]);
            // }
            // printf("\n");
            // free(q_host);
            // float* k_host = (float*)malloc(head_dim * p->n_kv_heads * sizeof(float));
            // CHECK_HIP(hipMemcpy(k_host, s->k, head_dim * p->n_kv_heads * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("apply_rotary_emb_k: "); for (int i = 0; i < 20; i++)
            // {
            //     printf("%f ", k_host[i]);
            // }
            // printf("\n");
            // free(k_host);
        }

        // ! Multihead attention (all on GPU)
        // Add bounds checking for attention computation
        if (pos >= p->seq_len) {
            printf("ERROR: Position %d exceeds sequence length %d\n", pos, p->seq_len);
            // Clean up allocated buffers before returning
            if (cos_vals)
                CHECK_HIP(hipFree(cos_vals));
            if (sin_vals)
                CHECK_HIP(hipFree(sin_vals));
            return nullptr;
        }

        {
            // HipTimer timer("compute_attention_scores");
            compute_attention_scores_hip_device(s->q, s->key_cache + loff, s->att, s->mask, pos,
                                                p->seq_len, head_dim, kv_dim, kv_mul,
                                                p->sliding_window, l, p->n_attn_heads);
            // float* att_host = (float*)malloc(p->n_attn_heads * p->seq_len * sizeof(float));
            // CHECK_HIP(hipMemcpy(att_host, s->att, p->n_attn_heads * p->seq_len * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("compute_attention_scores: "); for (int i = 0; i <
            // 20; i++) {
            //     printf("%f ", att_host[i]);
            // }
            // printf("\n");
            // free(att_host);
        }

        {
            // HipTimer timer("add_attention_sink");
            add_attention_sink_hip_device(s->att, w->attn_sinks + l * p->n_attn_heads, pos,
                                          p->seq_len, p->n_attn_heads, l);
            // float* att_host = (float*)malloc(p->n_attn_heads * p->seq_len * sizeof(float));
            // CHECK_HIP(hipMemcpy(att_host, s->att, p->n_attn_heads * p->seq_len * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("add_attention_sink: "); for (int i = 0; i < 20; i++)
            // {
            //     printf("%f ", att_host[i]);
            // }
            // printf("\n");
            // free(att_host);
        }

        {
            // HipTimer timer("softmax_attention");
            softmax_attention_hip_device(s->att, pos, p->seq_len, p->n_attn_heads);
            // float* att_host = (float*)malloc(p->n_attn_heads * p->seq_len * sizeof(float));
            // CHECK_HIP(hipMemcpy(att_host, s->att, p->n_attn_heads * p->seq_len * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("softmax_attention: "); for (int i = 0; i < 20; i++)
            // {
            //     printf("%f ", att_host[i]);
            // }
            // printf("\n");
            // free(att_host);
        }

        {
            // HipTimer timer("weighted_value_accumulation");
            weighted_value_accumulation_hip_device(s->att, s->value_cache + loff, s->tb, pos,
                                                   p->seq_len, head_dim, kv_dim, kv_mul,
                                                   p->n_attn_heads);
            // float* tb_host = (float*)malloc(head_dim * p->n_attn_heads * sizeof(float));
            // CHECK_HIP(hipMemcpy(tb_host, s->tb, head_dim * p->n_attn_heads * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("weighted_value_accumulation: "); for (int i = 0; i <
            // 20; i++) {
            //     printf("%f ", tb_host[i]);
            // }
            // printf("\n");
            // free(tb_host);
        }

        // ! Output projection: FP16 weights -> FP32 computation
        __half* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        __half* b_o = w->b_o + 1ll * l * hidden_dim;
        {
            // HipTimer timer("matmul_o");
            matmul_hip_hybrid_device(s->tb2, s->tb, w_o, b_o, head_dim * p->n_attn_heads,
                                     hidden_dim);
            // float* tb2_host = (float*)malloc(hidden_dim * sizeof(float));
            // CHECK_HIP(hipMemcpy(tb2_host, s->tb2, hidden_dim * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("matmul_o: "); for (int i = 0; i < 20; i++) {
            //     printf("%f ", tb2_host[i]);
            // }
            // printf("\n");
            // free(tb2_host);
        }

        // Add bias and residual connection
        // vecaddvec_hip_hybrid_device(s->tb2, b_o, 1.0f, hidden_dim);
        {
            // HipTimer timer("residual_connection");
            vecaddvec_hip_device(x, s->tb2, 1.0f, hidden_dim);
            // float* x_host = (float*)malloc(hidden_dim * sizeof(float));
            // CHECK_HIP(hipMemcpy(x_host, x, hidden_dim * sizeof(float), hipMemcpyDeviceToHost));
            // printf("residual_connection: ");
            // for (int i = 0; i < 20; i++) {
            //     printf("%f ", x_host[i]);
            // }
            // printf("\n");
            // free(x_host);
        }

        // ! FFN RMSNorm
        {
            // HipTimer timer("rmsnorm_ffn");
            rmsnorm_hip_hybrid_device(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);
            // float* t_host = (float*)malloc(hidden_dim * sizeof(float));
            // CHECK_HIP(hipMemcpy(t_host, s->t, hidden_dim * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("rmsnorm_ffn: "); for (int i = 0; i < 20; i++) {
            //     printf("%f ", t_host[i]);
            // }
            // printf("\n");
            // free(t_host);
        }

        // ! MoE Router
        __half* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        __half* b_router = w->b_router + 1ll * l * n_experts;
        {
            // HipTimer timer("matmul_router");
            matmul_hip_hybrid_device(s->router_score, s->t, w_router, b_router, hidden_dim,
                                     n_experts);
            // float* router_score_host = (float*)malloc(n_experts * sizeof(float));
            // CHECK_HIP(hipMemcpy(router_score_host, s->router_score, n_experts * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("matmul_router: "); for (int i = 0; i < 20; i++) {
            //     printf("%f ", router_score_host[i]);
            // }
            // printf("\n");
            // free(router_score_host);
        }
        // vecaddvec_hip_hybrid_device(s->router_score, b_router, 1.0f, n_experts);
        {
            // HipTimer timer("topk");
            topk_hip_device(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);
            // float* topk_v_host = (float*)malloc(p->experts_per_token * sizeof(float));
            // CHECK_HIP(hipMemcpy(topk_v_host, s->topk_v, p->experts_per_token * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("topk: "); for (int i = 0; i < 20; i++) {
            //     printf("%f ", topk_v_host[i]);
            // }
            // printf("\n");
            // free(topk_v_host);
            // float* topk_i_host = (float*)malloc(p->experts_per_token * sizeof(float));
            // CHECK_HIP(hipMemcpy(topk_i_host, s->topk_i, p->experts_per_token * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("topk_i: "); for (int i = 0; i < 20; i++) {
            //     printf("%f ", topk_i_host[i]);
            // }
            // printf("\n");
            // free(topk_i_host);
        }
        {
            // HipTimer timer("softmax router");
            softmax_hip_device(s->topk_v, p->experts_per_token);
            // float* topk_v_host = (float*)malloc(p->experts_per_token * sizeof(float));
            // CHECK_HIP(hipMemcpy(topk_v_host, s->topk_v, p->experts_per_token * sizeof(float),
            // hipMemcpyDeviceToHost)); printf("softmax_router: "); for (int i = 0; i < 20; i++) {
            //     printf("%f ", topk_v_host[i]);
            // }
            // printf("\n");
            // free(topk_v_host);
        }

        // ! Expert processing (OPTIMIZED: ONLY PROCESS ACTIVE EXPERTS)
        CHECK_HIP(hipMemset(s->e_agg, 0, hidden_dim * sizeof(float)));

        // PERFORMANCE FIX: Only process experts that are actually in topk
        // Copy topk_indices to host ONCE per layer (not per expert)
        int topk_indices_host[16]; // Safe buffer size (max possible experts_per_token)
        float topk_weights_host[16];

        // Add bounds checking for safety
        int safe_experts_per_token = (p->experts_per_token > 16) ? 16 : p->experts_per_token;

        CHECK_HIP(hipMemcpy(topk_indices_host, s->topk_i, safe_experts_per_token * sizeof(int),
                            hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(topk_weights_host, s->topk_v, safe_experts_per_token * sizeof(float),
                            hipMemcpyDeviceToHost));

        // Process ONLY the active experts (typically 2 out of 8)
        for (int idx = 0; idx < safe_experts_per_token; idx++) {
            int e = topk_indices_host[idx];
            float expert_w = topk_weights_host[idx];

            // ! Linear 1 (Gated MLP): FP16 weights -> FP32 computation
            // Add safety checks for weight offset calculations
            long long w_mlp1_offset =
                1ll * (l * n_experts + e) * (2 * p->intermediate_dim) * hidden_dim;
            long long b_mlp1_offset = 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);

            __half* w_mlp1 = w->w_mlp1 + w_mlp1_offset;
            __half* b_mlp1 = w->b_mlp1 + b_mlp1_offset;

            {
                // HipTimer timer("matmul_mlp1");
                matmul_hip_hybrid_device(s->mlp1_out, s->t, w_mlp1, b_mlp1, hidden_dim,
                                         2 * p->intermediate_dim);
                float* mlp1_out_host = (float*)malloc(2 * p->intermediate_dim * sizeof(float));
                // CHECK_HIP(hipMemcpy(mlp1_out_host, s->mlp1_out, 2 * p->intermediate_dim *
                // sizeof(float), hipMemcpyDeviceToHost)); printf("matmul_mlp1: "); for (int i = 0;
                // i < 20; i++) {
                //     printf("%f ", mlp1_out_host[i]);
                // }
                // printf("\n");
                // free(mlp1_out_host);
            }
            // vecaddvec_hip_hybrid_device(s->mlp1_out, b_mlp1, 1.0f, 2 * p->intermediate_dim);

            // Split mlp1_out into gate and up using optimized GPU kernel
            // Add safety check for buffer sizes
            if (s->mlp1_out == nullptr || s->gate == nullptr || s->up == nullptr) {
                printf("ERROR: NULL buffer in split_gate_up\n");
                // Clean up allocated buffers before returning
                if (cos_vals)
                    CHECK_HIP(hipFree(cos_vals));
                if (sin_vals)
                    CHECK_HIP(hipFree(sin_vals));
                return nullptr;
            }
            split_gate_up_hip_device(s->mlp1_out, s->gate, s->up, p->intermediate_dim);

            // ! SwiGLU non-linearity - use device-only version
            const float alpha = 1.702f;
            {
                // HipTimer timer("swiglu");
                thaDNN_s_swiglu(s->gate, s->up, p->intermediate_dim, alpha, p->swiglu_limit);
                // float* gate_host = (float*)malloc(p->intermediate_dim * sizeof(float));
                // CHECK_HIP(hipMemcpy(gate_host, s->gate, p->intermediate_dim * sizeof(float),
                // hipMemcpyDeviceToHost)); printf("swiglu: "); for (int i = 0; i < 20; i++) {
                //     printf("%f ", gate_host[i]);
                // }
                // printf("\n");
                // free(gate_host);
                // float* up_host = (float*)malloc(p->intermediate_dim * sizeof(float));
                // CHECK_HIP(hipMemcpy(up_host, s->up, p->intermediate_dim * sizeof(float),
                // hipMemcpyDeviceToHost)); printf("swiglu_up: "); for (int i = 0; i < 20; i++) {
                //     printf("%f ", up_host[i]);
                // }
                // printf("\n");
                // free(up_host);
            }
            // Copy result back to gate_up buffer
            CHECK_HIP(hipMemcpy(s->gate_up, s->gate, p->intermediate_dim * sizeof(float),
                                hipMemcpyDeviceToDevice));

            // ! Final matmul (down project): FP16 weights -> FP32 computation
            // Add safety checks for weight offset calculations
            long long w_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim * p->intermediate_dim;
            long long b_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim;

            __half* w_mlp2 = w->w_mlp2 + w_mlp2_offset;
            __half* b_mlp2 = w->b_mlp2 + b_mlp2_offset;

            {
                // HipTimer timer("matmul_mlp2");
                matmul_hip_hybrid_device(s->tb2, s->gate_up, w_mlp2, b_mlp2, hidden_dim,
                                         p->intermediate_dim);
                // float* tb2_host = (float*)malloc(hidden_dim * sizeof(float));
                // CHECK_HIP(hipMemcpy(tb2_host, s->tb2, hidden_dim * sizeof(float),
                // hipMemcpyDeviceToHost)); printf("matmul_mlp2: "); for (int i = 0; i < 20; i++) {
                //     printf("%f ", tb2_host[i]);
                // }
                // printf("\n");
                // free(tb2_host);
            }
            // vecaddvec_hip_hybrid_device(s->tb2, b_mlp2, 1.0f, hidden_dim);

            // ! Aggregate expert using direct GPU operation
            {
                // HipTimer timer("aggregate_expert");
                vecaddvec_hip_device(s->e_agg, s->tb2, expert_w, hidden_dim);
                // float* e_agg_host = (float*)malloc(hidden_dim * sizeof(float));
                // CHECK_HIP(hipMemcpy(e_agg_host, s->e_agg, hidden_dim * sizeof(float),
                // hipMemcpyDeviceToHost)); printf("aggregate_expert: "); for (int i = 0; i < 20;
                // i++) {
                //     printf("%f ", e_agg_host[i]);
                // }
                // printf("\n");
                // free(e_agg_host);
            }
        }
        // ! Residual connection (before rms2 to after MoE)
        {
            // HipTimer timer("residual_connection_e_agg");
            vecaddvec_hip_device(x, s->e_agg, 1.0f, hidden_dim);
            // float* x_host = (float*)malloc(hidden_dim * sizeof(float));
            // CHECK_HIP(hipMemcpy(x_host, x, hidden_dim * sizeof(float), hipMemcpyDeviceToHost));
            // printf("residual_connection_e_agg: ");
            // for (int i = 0; i < 20; i++) {
            //     printf("%f ", x_host[i]);
            // }
            // printf("\n");
            // free(x_host);
        }

        // Single sync per layer for optimal performance
        CHECK_HIP(hipDeviceSynchronize());
    }

    // Add safety checks for final operations
    if (x == nullptr) {
        printf("ERROR: x buffer is NULL before final operations\n");
        // Clean up allocated buffers before returning
        if (cos_vals)
            CHECK_HIP(hipFree(cos_vals));
        if (sin_vals)
            CHECK_HIP(hipFree(sin_vals));
        return nullptr;
    }
    if (w->rms_out_w == nullptr) {
        printf("ERROR: rms_out_w is NULL\n");
        // Clean up allocated buffers before returning
        if (cos_vals)
            CHECK_HIP(hipFree(cos_vals));
        if (sin_vals)
            CHECK_HIP(hipFree(sin_vals));
        return nullptr;
    }
    if (s->logits == nullptr) {
        printf("ERROR: logits buffer is NULL\n");
        // Clean up allocated buffers before returning
        if (cos_vals)
            CHECK_HIP(hipFree(cos_vals));
        if (sin_vals)
            CHECK_HIP(hipFree(sin_vals));
        return nullptr;
    }
    if (w->out == nullptr) {
        printf("ERROR: output weight matrix is NULL\n");
        // Clean up allocated buffers before returning
        if (cos_vals)
            CHECK_HIP(hipFree(cos_vals));
        if (sin_vals)
            CHECK_HIP(hipFree(sin_vals));
        return nullptr;
    }
    {
        // HipTimer timer("rmsnorm_out");
        rmsnorm_hip_hybrid_device(x, x, w->rms_out_w, hidden_dim);
        // float* x_host = (float*)malloc(hidden_dim * sizeof(float));
        // CHECK_HIP(hipMemcpy(x_host, x, hidden_dim * sizeof(float), hipMemcpyDeviceToHost));
        // printf("rmsnorm_out: ");
        // for (int i = 0; i < 20; i++) {
        //     printf("%f ", x_host[i]);
        // }
        // printf("\n");
        // free(x_host);
    }

    // ! linear: classifier into logits
    {
        // HipTimer timer("matmul_logits");
        matmul_hip_hybrid_device(s->logits, x, w->out, nullptr, hidden_dim, p->vocab_size);
        // float* logits_host = (float*)malloc(p->vocab_size * sizeof(float));
        // CHECK_HIP(hipMemcpy(logits_host, s->logits, p->vocab_size * sizeof(float),
        // hipMemcpyDeviceToHost)); printf("matmul_logits: "); for (int i = 0; i < 20; i++) {
        //     printf("%f ", logits_host[i]);
        // }
        // printf("\n");
        // free(logits_host);
    }

    CHECK_HIP(hipDeviceSynchronize());

    if (cos_vals) {
        CHECK_HIP(hipFree(cos_vals));
    }
    if (sin_vals) {
        CHECK_HIP(hipFree(sin_vals));
    }

    fflush(stdout);

    return s->logits;
}
