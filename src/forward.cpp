#include "../include/model.hpp"
#include "hip/BLAS.hip"
#include "hip/attention.hip"
#include "hip/embed.hip"
#include "hip/matvec.hip"
#include "hip/moe.hip"
#include "hip/rmsnorm.hip"
#include "hip/rope.hip"
#include "hip/scheduler.hip"
#include "hip/softmax.hip"
#include "hip/split_qkv.hip"
#include "hip/topk.hip"
#include "hip/vecadd.hip"
#include "profiler.cpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#include "../include/pp_manager.hpp"

float* forward_stage(OssTransformerHybrid* shard, bool has_embed, bool has_out, int* tokens,
                     const int* pos_per_token_h, int batch_size, const int* batch_indices_h,
                     int B_stride);

// Single hidden vector hop between devices
static void pp_pass_x_between(PPManager* mgr, int from_idx, int to_idx) {
    auto& A = mgr->shards[from_idx];
    auto& B = mgr->shards[to_idx];
    size_t bytes = (size_t)mgr->hidden_dim * (size_t)mgr->batch_size * sizeof(float);

    CHECK_HIP(hipSetDevice(A.device_id));
    CHECK_HIP(hipStreamSynchronize(0));

    if (mgr->peer_ok[A.device_id][B.device_id]) {
        CHECK_HIP(hipSetDevice(B.device_id));
        CHECK_HIP(hipMemcpyPeerAsync(B.shard->state.x, B.device_id, A.shard->state.x, A.device_id,
                                     bytes, 0));
        CHECK_HIP(hipStreamSynchronize(0));
    } else {
        if (!mgr->host_bounce)
            CHECK_HIP(hipHostMalloc((void**)&mgr->host_bounce, bytes));
        CHECK_HIP(hipSetDevice(A.device_id));
        CHECK_HIP(hipMemcpy(mgr->host_bounce, A.shard->state.x, bytes, hipMemcpyDeviceToHost));
        CHECK_HIP(hipSetDevice(B.device_id));
        CHECK_HIP(hipMemcpy(B.shard->state.x, mgr->host_bounce, bytes, hipMemcpyHostToDevice));
    }
}

static float* forward_token_pipeline(OssTransformerHybrid* root, int* tokens,
                                     const int* pos_per_token_h, int batch_size,
                                     const int* batch_indices_h, int B_stride) {
    PPManager* mgr = root->pp;
    // Stage 0
    CHECK_HIP(hipSetDevice(mgr->shards[0].device_id));
    float* ret =
        forward_stage(mgr->shards[0].shard, mgr->shards[0].has_embed, mgr->shards[0].has_out,
                      tokens, pos_per_token_h, batch_size, batch_indices_h, B_stride);
    // Middle stages
    for (int p = 1; p < mgr->pp; ++p) {
        pp_pass_x_between(mgr, p - 1, p);
        CHECK_HIP(hipSetDevice(mgr->shards[p].device_id));
        ret = forward_stage(mgr->shards[p].shard, mgr->shards[p].has_embed, mgr->shards[p].has_out,
                            tokens, pos_per_token_h, batch_size, batch_indices_h, B_stride);
    }
    return ret; // logits on last stage
}

float* forward_stage(OssTransformerHybrid* shard, bool has_embed, bool has_out, int* tokens,
                     const int* pos_per_token_h, int batch_size, const int* batch_indices_h,
                     int B_stride) {
    hipEvent_t start_total, end_total;
    hipEvent_t start_section, end_section;
    hipEvent_t start_moe_sub, end_moe_sub;

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

    OssConfig* p = &shard->config;
    OssTransformerWeightsHalf* w = &shard->weights;
    OssRunState* s = &shard->state;

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
    if (has_embed) {
        embed(x, w->token_embedding_table, s->d_tokens, batch_size, hidden_dim);
    }
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
        rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);
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
        matvec(s->qkv, s->t, w_qkv, b_qkv, batch_size, hidden_dim,
               (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float qkv_ms;
            CHECK_HIP(hipEventElapsedTime(&qkv_ms, start_section, end_section));
            g_batch_timings.qkv_projection_time += qkv_ms;
        }

        // ! Split QKV
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        split_qkv(s->qkv, s->q, s->key_cache, s->value_cache, batch_size, head_dim, p->n_attn_heads,
                  p->n_kv_heads, l, s->d_pos_per_token, p->seq_len, s->d_batch_indices, B_stride,
                  /*stream=*/0, kv_dim);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float split_qkv_ms;
            CHECK_HIP(hipEventElapsedTime(&split_qkv_ms, start_section, end_section));
            g_batch_timings.split_qkv_time += split_qkv_ms;
        }

        // ! RoPE (Q and K)
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        {
            const float ntk_beta = 32.0f;
            const float ntk_alpha = 1.0f;
            rope_qk(s->q, s->key_cache, batch_size, p->n_attn_heads, p->n_kv_heads, head_dim,
                    p->seq_len, kv_dim, l, s->d_pos_per_token, s->d_batch_indices, B_stride,
                    p->rope_theta, p->rope_scaling_factor, p->initial_context_length, ntk_beta,
                    ntk_alpha,
                    /*stream=*/0);
        }
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float rope_ms;
            CHECK_HIP(hipEventElapsedTime(&rope_ms, start_section, end_section));
            g_batch_timings.rope_time += rope_ms;
        }

        // ! Attention
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        fa(s->q, (const void*)s->key_cache, (const void*)s->value_cache, s->mask, w->attn_sinks,
           s->tb, batch_size, p->seq_len, head_dim, kv_dim, kv_mul, p->sliding_window, (int)l,
           p->n_attn_heads, s->d_pos_per_token, s->d_batch_indices, (long long)B_stride,
           max_pos_in_batch, 0);

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
        matvec(s->tb2, s->tb, w_o, b_o, batch_size, head_dim * p->n_attn_heads, hidden_dim);
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
        vecadd(x, s->tb2, 1.0f, batch_size, hidden_dim);
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
        rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, batch_size, hidden_dim);
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
        matvec(s->router_score, s->t, w_router, b_router, batch_size, hidden_dim, n_experts);

        // ! TopK and Softmax
        topk(s->topk_v, s->topk_i, s->router_score, batch_size, n_experts, p->experts_per_token);

        softmax(s->topk_v, batch_size, p->experts_per_token);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float moe_routing_ms;
            CHECK_HIP(hipEventElapsedTime(&moe_routing_ms, start_section, end_section));
            g_batch_timings.moe_routing_time += moe_routing_ms;
        }

        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }

        // ! MoE Setup - Zero aggregation
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

        // ! Build assignments + counts
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

        // ! Exclusive scan on device -> expert_offsets[0..E]
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

        // ! Compact by expert
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

        // ! Gather inputs X -> x_by_expert
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

        // ! === Build work queue and read back {active_experts, max_Ne} ===
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_moe_sub, 0));
        }

        thread_local static int* d_work_queue = nullptr;
        thread_local static int d_work_queue_capacity = 0;
        thread_local static Int2* d_meta = nullptr;

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
            vecadd(x, s->e_agg, 1.0f, batch_size, hidden_dim);
        } else {
            CHECK_HIP(hipMemset(s->e_agg, 0, batch_size * hidden_dim * sizeof(float)));

            const long long mlp1_weight_stride = (2LL * p->intermediate_dim) * hidden_dim;
            const long long mlp1_bias_stride = (2LL * p->intermediate_dim);
            const long long mlp2_weight_stride = hidden_dim * (long long)p->intermediate_dim;
            const long long mlp2_bias_stride = hidden_dim;
            const float alpha = 1.702f;

            const int work_start = 0;
            const int work_count = active_experts;

            float mlp1_ms = 0.0f, swiglu_ms = 0.0f, mlp2_ms = 0.0f, scatter_ms = 0.0f;

            // ! MLP1: [Ne,H] x [H,2I] -> [Ne,2I]
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }
            moe_matvec(d_work_queue, work_start, work_count, s->x_by_expert, s->mlp1_by_expert,
                       w->w_mlp1 + 1ll * l * n_experts * mlp1_weight_stride,
                       w->b_mlp1 + 1ll * l * n_experts * mlp1_bias_stride, hidden_dim,
                       2 * p->intermediate_dim, mlp1_weight_stride, mlp1_bias_stride, max_Ne, 0);
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                CHECK_HIP(hipEventElapsedTime(&mlp1_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_mlp1_time += mlp1_ms;
            }

            // ! SwiGLU split + activation: [Ne,2I] -> [Ne,I]
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }
            moe_split_swiglu(d_work_queue, work_start, work_count, s->mlp1_by_expert,
                             s->gate_by_expert, p->intermediate_dim, alpha, p->swiglu_limit, max_Ne,
                             0);
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                CHECK_HIP(hipEventElapsedTime(&swiglu_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_swiglu_time += swiglu_ms;
            }

            // ! MLP2: [Ne,I] x [I,H] -> [Ne,H]
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }
            moe_matvec(d_work_queue, work_start, work_count, s->gate_by_expert, s->y_by_expert,
                       w->w_mlp2 + 1ll * l * n_experts * mlp2_weight_stride,
                       w->b_mlp2 + 1ll * l * n_experts * mlp2_bias_stride, p->intermediate_dim,
                       hidden_dim, mlp2_weight_stride, mlp2_bias_stride, max_Ne, 0);
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                CHECK_HIP(hipEventElapsedTime(&mlp2_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_mlp2_time += mlp2_ms;
            }

            // ! Scale & scatter into e_agg
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(start_moe_sub, 0));
            }
            moe_scale_scatter(d_work_queue, work_start, work_count, s->y_by_expert, s->tokens_flat,
                              s->weights_flat, s->e_agg, hidden_dim, batch_size, max_Ne, 0);
            if (g_enable_profiling) {
                CHECK_HIP(hipEventRecord(end_moe_sub, 0));
                CHECK_HIP(hipEventSynchronize(end_moe_sub));
                CHECK_HIP(hipEventElapsedTime(&scatter_ms, start_moe_sub, end_moe_sub));
                g_batch_timings.moe_scatter_time += scatter_ms;

                g_batch_timings.expert_processing_time +=
                    (mlp1_ms + swiglu_ms + mlp2_ms + scatter_ms);
            }
        }

        // ! Residual
        vecadd(x, s->e_agg, 1.0f, batch_size, hidden_dim);
    }

    if (has_out) {
        // ! RMSNorm
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(start_section, 0));
        }
        rmsnorm(x, x, w->rms_out_w, batch_size, hidden_dim);

        // ! Linear logits
        matvec(s->logits, x, w->out, nullptr, batch_size, hidden_dim, p->vocab_size);
        if (g_enable_profiling) {
            CHECK_HIP(hipEventRecord(end_section, 0));
            CHECK_HIP(hipEventSynchronize(end_section));
            float final_ops_ms;
            CHECK_HIP(hipEventElapsedTime(&final_ops_ms, start_section, end_section));
            g_batch_timings.final_ops_time += final_ops_ms;
        }
    }

    if (g_enable_profiling) {
        CHECK_HIP(hipEventRecord(end_total, 0));
        CHECK_HIP(hipEventSynchronize(end_total));
        float total_ms;
        CHECK_HIP(hipEventElapsedTime(&total_ms, start_total, end_total));
        g_batch_timings.total_time += total_ms;
        g_batch_timings.num_calls++;
        g_batch_timings.total_layers += p->n_layers;

        CHECK_HIP(hipEventDestroy(start_total));
        CHECK_HIP(hipEventDestroy(end_total));
        CHECK_HIP(hipEventDestroy(start_section));
        CHECK_HIP(hipEventDestroy(end_section));
        CHECK_HIP(hipEventDestroy(start_moe_sub));
        CHECK_HIP(hipEventDestroy(end_moe_sub));
    }

    return has_out ? s->logits : x;
}

float* forward(OssTransformerHybrid* transformer, int* tokens, const int* pos_per_token_h,
               int batch_size, const int* batch_indices_h, int B_stride) {
    if (transformer->pp != nullptr) {
        return forward_token_pipeline(transformer, tokens, pos_per_token_h, batch_size,
                                      batch_indices_h, B_stride);
    }
    return forward_stage(transformer, true, true, tokens, pos_per_token_h, batch_size,
                         batch_indices_h, B_stride);
}
