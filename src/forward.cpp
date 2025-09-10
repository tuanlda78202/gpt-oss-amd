#include "../include/model.hpp"
#include "hip/BLAS.hip"
#include "hip/attention.hip"
#include "hip/continuous_batch_shims.hip"
#include "hip/embed.hip"
#include "hip/matvec.hip"
#include "hip/moe.hip"
#include "hip/prim_add.hip"
#include "hip/rmsnorm.hip"
#include "hip/rope.hip"
#include "hip/scheduler.hip"
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

// Global profiling flag - off by default
static bool g_enable_profiling = false;

void set_profiling_enabled(bool enabled) { g_enable_profiling = enabled; }

// Detailed timing infrastructure with MoE breakdown
struct BatchForwardTimings {
    double setup_time;
    double embedding_time;
    double layer_rmsnorm_time;
    double qkv_projection_time;
    double split_qkv_time;
    double rope_time;
    double attention_time;
    double output_projection_time;
    double residual_time;
    double mlp_rmsnorm_time;
    double moe_routing_time;
    double expert_processing_time;

    // Detailed MoE Expert Processing Breakdown
    double moe_setup_time;       // memset operations
    double moe_assignment_time;  // build assignments and counting
    double moe_scan_time;        // exclusive scan
    double moe_compact_time;     // compact by expert
    double moe_gather_time;      // gather inputs
    double moe_queue_build_time; // work queue building + metadata
    double moe_mlp1_time;        // MLP1 matrix multiplication
    double moe_swiglu_time;      // SwiGLU activation
    double moe_mlp2_time;        // MLP2 matrix multiplication
    double moe_scatter_time;     // scale and scatter
    double moe_sync_time;        // stream synchronization

    double final_ops_time;
    double total_time;

    int num_calls;
    int total_layers;

    BatchForwardTimings() { reset(); }

    void reset() {
        setup_time = embedding_time = layer_rmsnorm_time = qkv_projection_time = 0.0;
        split_qkv_time = rope_time = attention_time = output_projection_time = 0.0;
        residual_time = mlp_rmsnorm_time = moe_routing_time = expert_processing_time = 0.0;

        // Reset detailed MoE timings
        moe_setup_time = moe_assignment_time = moe_scan_time = moe_compact_time = 0.0;
        moe_gather_time = moe_queue_build_time = moe_mlp1_time = moe_swiglu_time = 0.0;
        moe_mlp2_time = moe_scatter_time = moe_sync_time = 0.0;

        final_ops_time = total_time = 0.0;
        num_calls = total_layers = 0;
    }

    void print_summary() {
        if (num_calls == 0)
            return;

        printf("%.*s\n", 70,
               "======================================================================");
        printf("BATCH FORWARD TIMING SUMMARY (%d calls, %d total layers)\n", num_calls,
               total_layers);
        printf("%.*s\n", 70,
               "======================================================================");

        // Main sections
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Setup & Memory", setup_time / num_calls,
               100.0 * setup_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Embedding", embedding_time / num_calls,
               100.0 * embedding_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Layer RMSNorm", layer_rmsnorm_time / num_calls,
               100.0 * layer_rmsnorm_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "QKV Projection", qkv_projection_time / num_calls,
               100.0 * qkv_projection_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Split QKV", split_qkv_time / num_calls,
               100.0 * split_qkv_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "RoPE", rope_time / num_calls,
               100.0 * rope_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Attention", attention_time / num_calls,
               100.0 * attention_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Output Projection",
               output_projection_time / num_calls, 100.0 * output_projection_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Residual", residual_time / num_calls,
               100.0 * residual_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "MLP RMSNorm", mlp_rmsnorm_time / num_calls,
               100.0 * mlp_rmsnorm_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "MoE Routing", moe_routing_time / num_calls,
               100.0 * moe_routing_time / total_time);

        // Expert Processing total
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Expert Processing [TOTAL]",
               expert_processing_time / num_calls, 100.0 * expert_processing_time / total_time);

        // Detailed MoE Expert Processing Breakdown
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "MoE Setup/Memset", moe_setup_time / num_calls,
               100.0 * moe_setup_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Assignment/Counting",
               moe_assignment_time / num_calls, 100.0 * moe_assignment_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Exclusive Scan", moe_scan_time / num_calls,
               100.0 * moe_scan_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Compact by Expert", moe_compact_time / num_calls,
               100.0 * moe_compact_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Gather Inputs", moe_gather_time / num_calls,
               100.0 * moe_gather_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Queue Build/Meta",
               moe_queue_build_time / num_calls, 100.0 * moe_queue_build_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "MLP1 MatVec", moe_mlp1_time / num_calls,
               100.0 * moe_mlp1_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "SwiGLU Activation", moe_swiglu_time / num_calls,
               100.0 * moe_swiglu_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "MLP2 MatVec", moe_mlp2_time / num_calls,
               100.0 * moe_mlp2_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Scale/Scatter", moe_scatter_time / num_calls,
               100.0 * moe_scatter_time / total_time);
        printf("  └─ %-21s: %8.3f ms (%.1f%%)\n", "Stream Sync", moe_sync_time / num_calls,
               100.0 * moe_sync_time / total_time);

        printf("%-25s: %8.3f ms (%.1f%%)\n", "Final Operations", final_ops_time / num_calls,
               100.0 * final_ops_time / total_time);

        printf("%.*s\n", 70,
               "======================================================================");
        printf("%-25s: %8.3f ms\n", "TOTAL", total_time / num_calls);
        printf("%.*s\n", 70,
               "======================================================================");
        fflush(stdout);
    }
};

// Global timing instance
static BatchForwardTimings g_batch_timings;

void reset_batch_timings() {
    if (g_enable_profiling) {
        g_batch_timings.reset();
    }
}

void print_batch_timing_summary() {
    if (g_enable_profiling) {
        g_batch_timings.print_summary();
    }
}

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

            // ! Final matmul (down project)
            long long w_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim * p->intermediate_dim;
            long long b_mlp2_offset = 1ll * (l * n_experts + e) * hidden_dim;

            __half* w_mlp2 = w->w_mlp2 + w_mlp2_offset;
            __half* b_mlp2 = w->b_mlp2 + b_mlp2_offset;

            matvec_gpu(s->tb2, s->gate, w_mlp2, b_mlp2, hidden_dim, p->intermediate_dim);

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

float* forward_hybrid_batch(OssTransformerHybrid* transformer, int* tokens,
                            const int* pos_per_token_h, int batch_size, const int* batch_indices_h,
                            int B_stride) {
    // Create timing events only if profiling is enabled
    hipEvent_t start_total, end_total;
    hipEvent_t start_section, end_section;
    hipEvent_t start_moe_sub, end_moe_sub; // For detailed MoE breakdown

    if (g_enable_profiling) {
        CHECK_HIP(hipEventCreate(&start_total));
        CHECK_HIP(hipEventCreate(&end_total));
        CHECK_HIP(hipEventCreate(&start_section));
        CHECK_HIP(hipEventCreate(&end_section));
        CHECK_HIP(hipEventCreate(&start_moe_sub));
        CHECK_HIP(hipEventCreate(&end_moe_sub));

        CHECK_HIP(hipEventRecord(start_total, 0));
        CHECK_HIP(hipEventRecord(start_section, 0));
    }

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

    int max_pos_in_batch = 0;
    for (int i = 0; i < batch_size; ++i)
        if (pos_per_token_h[i] > max_pos_in_batch)
            max_pos_in_batch = pos_per_token_h[i];

    CHECK_HIP(hipMemcpy(s->d_batch_indices, batch_indices_h, batch_size * sizeof(int),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s->d_tokens, tokens, batch_size * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(s->d_pos_per_token, pos_per_token_h, batch_size * sizeof(int),
                        hipMemcpyHostToDevice));

    // Record setup time
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_section, 0));
        CHECK_HIP(hipEventSynchronize(end_section));
        float setup_ms;
        CHECK_HIP(hipEventElapsedTime(&setup_ms, start_section, end_section));
        g_batch_timings.setup_time += setup_ms;
    }

    // ! Embedding
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(start_section, 0));
    }
    embed_batch_gpu(x, w->token_embedding_table, s->d_tokens, batch_size, hidden_dim);
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_section, 0));
        CHECK_HIP(hipEventSynchronize(end_section));
        float embedding_ms;
        CHECK_HIP(hipEventElapsedTime(&embedding_ms, start_section, end_section));
        g_batch_timings.embedding_time += embedding_ms;
    }

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // ! RMSNorm
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        rmsnorm_batch_gpu(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float rmsnorm_ms;
            CHECK_HIP(hipEventElapsedTime(&rmsnorm_ms, start_section, end_section));
            g_batch_timings.layer_rmsnorm_time += rmsnorm_ms;
        }

        // ! QKV project
        __half* w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                       (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        __half* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        matvec_batch_gpu(s->qkv, s->t, w_qkv, b_qkv, batch_size, hidden_dim,
                         (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float qkv_ms;
            CHECK_HIP(hipEventElapsedTime(&qkv_ms, start_section, end_section));
            g_batch_timings.qkv_projection_time += qkv_ms;
        }

        // ! Split QKV -> write KV at each token's own pos
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        split_qkv_batch_gpu_mixedpos(s->qkv, s->q, s->key_cache, s->value_cache, batch_size,
                                     head_dim, p->n_attn_heads, p->n_kv_heads, l,
                                     s->d_pos_per_token, p->seq_len, s->d_batch_indices, B_stride,
                                     /*stream=*/0, kv_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float split_qkv_ms;
            CHECK_HIP(hipEventElapsedTime(&split_qkv_ms, start_section, end_section));
            g_batch_timings.split_qkv_time += split_qkv_ms;
        }

        // ! RoPE (Q and K) per token
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        {
            const float ntk_beta = 32.0f;
            const float ntk_alpha = 1.0f;
            rope_qk_fused_batch_gpu_mixed(s->q, s->key_cache, batch_size, p->n_attn_heads,
                                          p->n_kv_heads, head_dim, p->seq_len, kv_dim, l,
                                          s->d_pos_per_token, s->d_batch_indices, B_stride,
                                          p->rope_theta, p->rope_scaling_factor,
                                          p->initial_context_length, ntk_beta, ntk_alpha,
                                          /*stream=*/0);
        }
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float rope_ms;
            CHECK_HIP(hipEventElapsedTime(&rope_ms, start_section, end_section));
            g_batch_timings.rope_time += rope_ms;
        }

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        flash_attn_decode_gpu_batch_mixed(
            /*q_batch=*/s->q,                      // (B, H*D)
            /*k_cache=*/(const void*)s->key_cache, // base; wrapper does layer/slot/pos math
            /*v_cache=*/(const void*)s->value_cache,
            /*mask=*/s->mask,             // (T, T)
            /*attn_sinks=*/w->attn_sinks, // pass base; wrapper slices by layer
            /*tb_batch=*/s->tb,           // (B, H*D) output
            /*B=*/batch_size,
            /*seq_len=*/p->seq_len,
            /*head_dim=*/head_dim,
            /*kv_dim=*/kv_dim,
            /*kv_mul=*/kv_mul,
            /*sliding_window=*/p->sliding_window,
            /*layer_idx=*/(int)l,
            /*n_attn_heads=*/p->n_attn_heads,
            /*d_pos_per_token=*/s->d_pos_per_token, // (B) device
            /*d_batch_indices=*/s->d_batch_indices, // (B) device
            /*B_stride=*/(long long)B_stride,       // elements per (B*T*kv_dim) per layer
            /*max_pos_in_batch=*/max_pos_in_batch,  // from host
            /*stream=*/0);                          // ✅ (fixed typo)
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float attn_ms;
            CHECK_HIP(hipEventElapsedTime(&attn_ms, start_section, end_section));
            g_batch_timings.attention_time += attn_ms;
        }

        // ! Output projection
        __half* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        __half* b_o = w->b_o + 1ll * l * hidden_dim;

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        matvec_batch_gpu(s->tb2, s->tb, w_o, b_o, batch_size, head_dim * p->n_attn_heads,
                         hidden_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float out_proj_ms;
            CHECK_HIP(hipEventElapsedTime(&out_proj_ms, start_section, end_section));
            g_batch_timings.output_projection_time += out_proj_ms;
        }

        // ! Residual
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        vec_add_vec_batch_gpu(x, s->tb2, 1.0f, batch_size, hidden_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float residual_ms;
            CHECK_HIP(hipEventElapsedTime(&residual_ms, start_section, end_section));
            g_batch_timings.residual_time += residual_ms;
        }

        // ! RMSNorm
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        rmsnorm_batch_gpu(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float mlp_rmsnorm_ms;
            CHECK_HIP(hipEventElapsedTime(&mlp_rmsnorm_ms, start_section, end_section));
            g_batch_timings.mlp_rmsnorm_time += mlp_rmsnorm_ms;
        }

        // ! MoE Router
        __half* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        __half* b_router = w->b_router + 1ll * l * n_experts;

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        matvec_batch_gpu(s->router_score, s->t, w_router, b_router, batch_size, hidden_dim,
                         n_experts);

        // ! TopK and Softmax
        topk_batch_gpu(s->topk_v, s->topk_i, s->router_score, batch_size, n_experts,
                       p->experts_per_token);

        softmax_batch_gpu(s->topk_v, batch_size, p->experts_per_token);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float moe_routing_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_routing_ms, start_section, end_section));
            g_batch_timings.moe_routing_time += moe_routing_ms;
        }

        // Expert processing timing (with detailed breakdown)
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }

        // MoE Setup - Zero aggregation
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        CHECK_HIP(hipMemset(s->e_agg, 0, batch_size * hidden_dim * sizeof(float)));
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_setup_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_setup_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_setup_time += moe_setup_ms;
        }

        const int BKE = batch_size * p->experts_per_token;
        dim3 blk1(256), grd1((BKE + blk1.x - 1) / blk1.x);

        // Build assignments + counts
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        CHECK_HIP(hipMemset(s->expert_counts, 0, n_experts * sizeof(int)));
        hipLaunchKernelGGL(build_assignments_and_count_kernel, grd1, blk1, 0, 0, s->topk_i,
                           s->topk_v, batch_size, p->experts_per_token, s->assign_expert,
                           s->assign_token, s->assign_weight, s->expert_counts, n_experts);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_assignment_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_assignment_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_assignment_time += moe_assignment_ms;
        }

        // Exclusive scan on device -> expert_offsets[0..E]
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        hipLaunchKernelGGL(exclusive_scan_small_kernel, dim3(1), dim3(1), 0, 0, s->expert_counts,
                           s->expert_offsets, n_experts);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_scan_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_scan_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_scan_time += moe_scan_ms;
        }

        // Compact by expert
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        CHECK_HIP(hipMemset(s->expert_counts, 0, n_experts * sizeof(int)));
        hipLaunchKernelGGL(compact_by_expert_kernel, grd1, blk1, 0, 0, s->assign_expert,
                           s->assign_token, s->assign_weight, BKE, s->expert_offsets,
                           s->expert_counts, s->tokens_flat, s->weights_flat);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_compact_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_compact_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_compact_time += moe_compact_ms;
        }

        // Gather inputs X -> x_by_expert
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }
        {
            const bool vecOK = ((reinterpret_cast<uintptr_t>(s->t) & 0xF) == 0) &&
                               ((reinterpret_cast<uintptr_t>(s->x_by_expert) & 0xF) == 0) &&
                               (hidden_dim % 4 == 0);
            if (vecOK) {
                const int H4 = hidden_dim >> 2;
                dim3 blk(256, 1, 1), grd((H4 + 255) / 256, BKE);
                hipLaunchKernelGGL(gather_rows_vec4_kernel, grd, blk, 0, 0, s->t, s->tokens_flat,
                                   s->x_by_expert, BKE, H4);
            } else {
                dim3 blk(256, 1, 1), grd((hidden_dim + 255) / 256, BKE);
                hipLaunchKernelGGL(gather_rows_kernel, grd, blk, 0, 0, s->t, s->tokens_flat,
                                   s->x_by_expert, BKE, hidden_dim);
            }
        }
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_gather_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_gather_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_gather_time += moe_gather_ms;
        }

        // === Build work queue and read back {active_experts, max_Ne} ===
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }

        static int* d_work_queue = nullptr;
        static int d_work_queue_capacity = 0; // ints
        static Int2* d_meta = nullptr;

        const int required_ints = 3 * n_experts;
        if (required_ints > d_work_queue_capacity) {
            if (d_work_queue)
                CHECK_HIP(hipFree(d_work_queue));
            CHECK_HIP(hipMalloc(&d_work_queue, required_ints * sizeof(int)));
            d_work_queue_capacity = required_ints;
        }
        if (!d_meta)
            CHECK_HIP(hipMalloc(&d_meta, sizeof(Int2)));

        hipLaunchKernelGGL(build_expert_work_queue_kernel, dim3(1), dim3(1), 0, 0,
                           s->expert_offsets, d_work_queue, d_meta, n_experts);

        Int2 h_meta;
        CHECK_HIP(hipMemcpy(&h_meta, d_meta, sizeof(Int2), hipMemcpyDeviceToHost));
        const int active_experts = h_meta.x;
        const int max_Ne = h_meta.y;

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_moe_sub, 0));
            CHECK_HIP(hipEventSynchronize(end_moe_sub));
            float moe_queue_build_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_queue_build_ms, start_moe_sub, end_moe_sub));
            g_batch_timings.moe_queue_build_time += moe_queue_build_ms;
        }

        if (active_experts == 0) {
            vec_add_vec_batch_gpu(x, s->e_agg, 1.0f, batch_size, hidden_dim);
        } else {
            // Multi-stream parallel processing
            constexpr int NUM_STREAMS = 4;
            static hipStream_t expert_streams[NUM_STREAMS];
            static bool streams_created = false;
            if (!streams_created) {
                for (int i = 0; i < NUM_STREAMS; ++i)
                    CHECK_HIP(hipStreamCreate(&expert_streams[i]));
                streams_created = true;
            }
            const int experts_per_stream = (active_experts + NUM_STREAMS - 1) / NUM_STREAMS;

            // Strides
            const long long mlp1_weight_stride = (2LL * p->intermediate_dim) * hidden_dim;
            const long long mlp1_bias_stride = (2LL * p->intermediate_dim);
            const long long mlp2_weight_stride = hidden_dim * (long long)p->intermediate_dim;
            const long long mlp2_bias_stride = hidden_dim;
            const float alpha = 1.702f;

            // MLP1 Processing
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }

            for (int sid = 0; sid < NUM_STREAMS; ++sid) {
                const int work_start = sid * experts_per_stream;
                const int work_end = std::min(work_start + experts_per_stream, active_experts);
                if (work_start >= work_end)
                    break;
                const int work_count = work_end - work_start;
                hipStream_t stream = expert_streams[sid];

                // ! MLP1: [Ne,H] x [H,2I] -> [Ne,2I]
                multi_expert_matvec_gpu(
                    d_work_queue, work_start, work_count, s->x_by_expert, s->mlp1_by_expert,
                    w->w_mlp1 + 1ll * l * n_experts * mlp1_weight_stride,
                    w->b_mlp1 + 1ll * l * n_experts * mlp1_bias_stride, hidden_dim,
                    2 * p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride, max_Ne, stream);
            }

            for (int i = 0; i < NUM_STREAMS; ++i)
                CHECK_HIP(hipStreamSynchronize(expert_streams[i]));

            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                float moe_mlp1_ms;
                CHECK_HIP(hipEventElapsedTime(&moe_mlp1_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_mlp1_time += moe_mlp1_ms;
            }

            // SwiGLU Processing
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }

            for (int sid = 0; sid < NUM_STREAMS; ++sid) {
                const int work_start = sid * experts_per_stream;
                const int work_end = std::min(work_start + experts_per_stream, active_experts);
                if (work_start >= work_end)
                    break;
                const int work_count = work_end - work_start;
                hipStream_t stream = expert_streams[sid];

                // ! split + SwiGLU: [Ne,2I] -> [Ne,I]
                multi_expert_split_swiglu_gpu(
                    d_work_queue, work_start, work_count, s->mlp1_by_expert, s->gate_by_expert,
                    p->intermediate_dim, alpha, p->swiglu_limit, max_Ne, stream);
            }

            for (int i = 0; i < NUM_STREAMS; ++i)
                CHECK_HIP(hipStreamSynchronize(expert_streams[i]));

            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                float moe_swiglu_ms;
                CHECK_HIP(hipEventElapsedTime(&moe_swiglu_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_swiglu_time += moe_swiglu_ms;
            }

            // MLP2 Processing
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }

            for (int sid = 0; sid < NUM_STREAMS; ++sid) {
                const int work_start = sid * experts_per_stream;
                const int work_end = std::min(work_start + experts_per_stream, active_experts);
                if (work_start >= work_end)
                    break;
                const int work_count = work_end - work_start;
                hipStream_t stream = expert_streams[sid];

                // ! MLP2: [Ne,I] x [I,H] -> [Ne,H]
                multi_expert_matvec_gpu(
                    d_work_queue, work_start, work_count, s->gate_by_expert, s->y_by_expert,
                    w->w_mlp2 + 1ll * l * n_experts * mlp2_weight_stride,
                    w->b_mlp2 + 1ll * l * n_experts * mlp2_bias_stride, p->intermediate_dim,
                    hidden_dim, mlp2_weight_stride, mlp2_bias_stride, max_Ne, stream);
            }

            for (int i = 0; i < NUM_STREAMS; ++i)
                CHECK_HIP(hipStreamSynchronize(expert_streams[i]));

            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                float moe_mlp2_ms;
                CHECK_HIP(hipEventElapsedTime(&moe_mlp2_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_mlp2_time += moe_mlp2_ms;
            }

            // Scale and Scatter Processing
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }

            for (int sid = 0; sid < NUM_STREAMS; ++sid) {
                const int work_start = sid * experts_per_stream;
                const int work_end = std::min(work_start + experts_per_stream, active_experts);
                if (work_start >= work_end)
                    break;
                const int work_count = work_end - work_start;
                hipStream_t stream = expert_streams[sid];

                // ! scale + scatter -> e_agg
                multi_expert_scale_scatter_gpu(d_work_queue, work_start, work_count, s->y_by_expert,
                                               s->tokens_flat, s->weights_flat, s->e_agg,
                                               hidden_dim, batch_size, max_Ne, stream);
            }

            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                float moe_scatter_ms;
                CHECK_HIP(hipEventElapsedTime(&moe_scatter_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_scatter_time += moe_scatter_ms;
            }

            // Final Stream Synchronization
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }

            for (int i = 0; i < NUM_STREAMS; ++i)
                CHECK_HIP(hipStreamSynchronize(expert_streams[i]));

            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                float moe_sync_ms;
                CHECK_HIP(hipEventElapsedTime(&moe_sync_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_sync_time += moe_sync_ms;
            }
        }

        // End expert processing timing
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float expert_processing_ms;
            CHECK_HIP(hipEventElapsedTime(&expert_processing_ms, start_section, end_section));
            g_batch_timings.expert_processing_time += expert_processing_ms;
        }

        // Residual connection - batched
        vec_add_vec_batch_gpu(x, s->e_agg, 1.0f, batch_size, hidden_dim);
    }

    // Final operations - batched
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(start_section, 0));
    }
    rmsnorm_batch_gpu(x, x, w->rms_out_w, batch_size, hidden_dim);

    // Linear: classifier into logits - batched
    matvec_batch_gpu(s->logits, x, w->out, nullptr, batch_size, hidden_dim, p->vocab_size);
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_section, 0));
        CHECK_HIP(hipEventSynchronize(end_section));
        float final_ops_ms;
        CHECK_HIP(hipEventElapsedTime(&final_ops_ms, start_section, end_section));
        g_batch_timings.final_ops_time += final_ops_ms;
    }

    // Record total time and cleanup events
    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_total, 0));
        CHECK_HIP(hipEventSynchronize(end_total));
        float total_ms;
        CHECK_HIP(hipEventElapsedTime(&total_ms, start_total, end_total));
        g_batch_timings.total_time += total_ms;
        g_batch_timings.num_calls++;
        g_batch_timings.total_layers += p->n_layers;

        // Cleanup events
        CHECK_HIP(hipEventDestroy(start_total));
        CHECK_HIP(hipEventDestroy(end_total));
        CHECK_HIP(hipEventDestroy(start_section));
        CHECK_HIP(hipEventDestroy(end_section));
        CHECK_HIP(hipEventDestroy(start_moe_sub));
        CHECK_HIP(hipEventDestroy(end_moe_sub));
    }

    return s->logits;
}
