#define GETP_SKIP_TYPEDEFS
#include "../../include/tokenizer.hpp"
#include "../forward_120b.cpp"
#include "../model_120b.cpp"
#include "../sampler.cpp"
#include "eval.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <omp.h>
#include <stdbool.h>

#ifndef GETP_RUN
#define GETP_RUN

static OssTransformerHybrid** g_all_models = nullptr;   // per-GPU transformer instances
static OssExpertShard* g_expert_shards = nullptr;       // expert shards shared across DP
static OssExpertParallelGroup* g_dp_groups = nullptr;   // data-parallel groups
thread_local OssExpertParallelGroup* t_group = nullptr; // thread-local group handle
thread_local OssTransformerHybrid* t_d = nullptr;       // primary shard for this thread

static int g_dp_world_size = 1;
static int g_ep_size = 1;
static int g_active_devices = 0;

void warm_up(Transformer* transformer, Tokenizer* tokenizer, int batch_size, int use_kv16) {
    OssTransformer* transformer_oss = (OssTransformer*)transformer;
    transformer_oss->config.batch_size = batch_size;

    int available_devices = 0;
    CHECK_HIP(hipGetDeviceCount(&available_devices));

    // const char* dp_env = getenv("GETP_DP");
    // const char* ep_env = getenv("GETP_EP");
    // int requested_dp = dp_env ? atoi(dp_env) : 1;
    // int requested_ep = ep_env ? atoi(ep_env) : 1;
    int requested_dp = 8;
    int requested_ep = 8;
    if (requested_dp <= 0)
        requested_dp = 1;
    if (requested_ep <= 0)
        requested_ep = 1;

    if (requested_ep > available_devices)
        requested_ep = available_devices;
    g_ep_size = requested_ep;
    g_dp_world_size = (requested_dp > available_devices) ? available_devices : requested_dp;
    int required_devices = std::max(g_dp_world_size, g_ep_size);
    if (required_devices > available_devices) {
        fprintf(stderr, "Requested DP=%d/EP=%d requires %d GPUs but only %d available\n",
                g_dp_world_size, g_ep_size, required_devices, available_devices);
        exit(EXIT_FAILURE);
    }

    printf("\n[Parallel Config] dp=%d, ep=%d, devices=%d (available=%d), batch_size=%d\n",
           g_dp_world_size, g_ep_size, required_devices, available_devices, batch_size);

    g_all_models = (OssTransformerHybrid**)malloc(sizeof(OssTransformerHybrid*) * required_devices);
    if (!g_all_models) {
        fprintf(stderr, "malloc g_all_models failed\n");
        exit(EXIT_FAILURE);
    }
    g_dp_groups = (OssExpertParallelGroup*)malloc(sizeof(OssExpertParallelGroup) * g_dp_world_size);
    if (!g_dp_groups) {
        fprintf(stderr, "malloc g_dp_groups failed\n");
        exit(EXIT_FAILURE);
    }

    // Enable peer access between all participating devices when possible
    for (int src = 0; src < required_devices; ++src) {
        CHECK_HIP(hipSetDevice(src));
        for (int dst = 0; dst < required_devices; ++dst) {
            if (src == dst)
                continue;
            int can_access = 0;
            CHECK_HIP(hipDeviceCanAccessPeer(&can_access, src, dst));
            if (can_access) {
                hipError_t perr = hipDeviceEnablePeerAccess(dst, 0);
                if (perr != hipSuccess && perr != hipErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "hipDeviceEnablePeerAccess(%d,%d) failed: %s\n", src, dst,
                            hipGetErrorString(perr));
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

#pragma omp parallel for num_threads(required_devices) schedule(static)
    for (int idx = 0; idx < required_devices; ++idx) {
        int device_id = idx;
        int dp_rank = (idx < g_dp_world_size) ? idx : -1;
        int ep_rank = (idx < g_ep_size) ? idx : -1;

        CHECK_HIP(hipSetDevice(device_id));

        g_all_models[idx] = (OssTransformerHybrid*)malloc(sizeof(OssTransformerHybrid));
        if (!g_all_models[idx]) {
            fprintf(stderr, "malloc model for device %d failed\n", device_id);
            exit(EXIT_FAILURE);
        }

        copy_transformer_to_device(transformer_oss, g_all_models[idx], device_id, dp_rank,
                                          g_ep_size, ep_rank, use_kv16);

        size_t free_mem, total_mem;
        CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
#pragma omp critical
        {
            printf("\n--- HYBRID WARM-UP COMPLETE (device %d | dp=%d ep=%d) ---\n", device_id,
                   dp_rank, ep_rank);
            printf("GPU Memory Status: Total %.2f GB, Used %.2f GB, Free %.2f GB\n",
                   total_mem / (1024.0 * 1024.0 * 1024.0),
                   (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0),
                   free_mem / (1024.0 * 1024.0 * 1024.0));
            printf("-----------------------------------------------\n");
        }
    }

    g_active_devices = required_devices;

    if (g_ep_size > 0) {
        g_expert_shards = (OssExpertShard*)malloc(sizeof(OssExpertShard) * g_ep_size);
        if (!g_expert_shards) {
            fprintf(stderr, "malloc expert shards failed\n");
            exit(EXIT_FAILURE);
        }
        for (int ep = 0; ep < g_ep_size; ++ep) {
            g_expert_shards[ep].model = g_all_models[ep];
            g_expert_shards[ep].device_id = ep;
            g_expert_shards[ep].workspace.x_by_expert = nullptr;
            g_expert_shards[ep].workspace.mlp1_by_expert = nullptr;
            g_expert_shards[ep].workspace.gate_by_expert = nullptr;
            g_expert_shards[ep].workspace.y_by_expert = nullptr;
            g_expert_shards[ep].workspace.capacity_tokens = 0;
            g_expert_shards[ep].mutex_handle = new std::mutex();
        }
    }

    for (int dp = 0; dp < g_dp_world_size; ++dp) {
        g_dp_groups[dp].dp_rank = dp;
        g_dp_groups[dp].ep_size = g_ep_size;
        g_dp_groups[dp].primary_shard_index = (dp < g_ep_size) ? dp : -1;
        if (g_ep_size > 0) {
            g_dp_groups[dp].shards = (OssExpertShard**)malloc(sizeof(OssExpertShard*) * g_ep_size);
            if (!g_dp_groups[dp].shards) {
                fprintf(stderr, "malloc shards pointers for dp=%d failed\n", dp);
                exit(EXIT_FAILURE);
            }
            for (int ep = 0; ep < g_ep_size; ++ep) {
                g_dp_groups[dp].shards[ep] = &g_expert_shards[ep];
            }
        } else {
            g_dp_groups[dp].shards = nullptr;
        }
    }

    omp_set_num_threads(g_dp_world_size);

    reset_batch_timings();
}

void finish(Transformer* transformer, Tokenizer* tokenizer) {
    print_batch_timing_summary();

    if (g_dp_groups) {
        for (int dp = 0; dp < g_dp_world_size; ++dp) {
            if (g_dp_groups[dp].shards) {
                free(g_dp_groups[dp].shards);
                g_dp_groups[dp].shards = nullptr;
            }
        }
        free(g_dp_groups);
        g_dp_groups = nullptr;
    }

    if (g_expert_shards) {
        for (int ep = 0; ep < g_ep_size; ++ep) {
            OssExpertShard* shard = &g_expert_shards[ep];
            CHECK_HIP(hipSetDevice(shard->device_id));
            if (shard->workspace.x_by_expert)
                CHECK_HIP(hipFree(shard->workspace.x_by_expert));
            if (shard->workspace.mlp1_by_expert)
                CHECK_HIP(hipFree(shard->workspace.mlp1_by_expert));
            if (shard->workspace.gate_by_expert)
                CHECK_HIP(hipFree(shard->workspace.gate_by_expert));
            if (shard->workspace.y_by_expert)
                CHECK_HIP(hipFree(shard->workspace.y_by_expert));
            delete reinterpret_cast<std::mutex*>(shard->mutex_handle);
            shard->mutex_handle = nullptr;
        }
        free(g_expert_shards);
        g_expert_shards = nullptr;
    }

    if (g_all_models) {
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < g_active_devices; ++idx) {
            CHECK_HIP(hipSetDevice(idx));
            if (g_all_models[idx]) {
                free_transformer_on_device_hybrid(g_all_models[idx]);
                free(g_all_models[idx]);
            }
        }
        free(g_all_models);
        g_all_models = nullptr;
    }

    g_active_devices = 0;

    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    size_t used_mem = total_mem - free_mem;
    printf("\n--- HYBRID FINISH COMPLETE ---\n");
    printf("GPU Memory Status:\n");
    printf("  Total: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Used: %.2f GB\n", used_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Free: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
    printf("-------------------------------\n");
}

void clear_lines(int num_lines) {
    for (int i = 0; i < num_lines; i++) {
        printf("\033[A\033[K");
    }
    fflush(stdout);
}

void progress_bar(int batch_size, int* tokens_generated, int* max_tokens, bool* finished) {
    const char* GREEN = "\033[32m";
    const char* BLUE = "\033[34m";
    const char* YELLOW = "\033[33m";
    const char* RED = "\033[31m";
    const char* RESET = "\033[0m";
    const char* BOLD = "\033[1m";

    for (int i = 0; i < batch_size; i++) {
        if (finished[i]) {
            printf("%s#%-2d %s●●●●●●●●●●●●●●●● %s✓ Done%s\n", BOLD, i + 1, GREEN, GREEN, RESET);
        } else {
            int progress = (max_tokens[i] > 0) ? (tokens_generated[i] * 16 / max_tokens[i]) : 0;
            if (progress > 16)
                progress = 16;

            printf("%s#%-2d %s", BOLD, i + 1, RESET);

            const char* progress_color;
            if (progress <= 4) {
                progress_color = RED;
            } else if (progress <= 8) {
                progress_color = YELLOW;
            } else if (progress <= 12) {
                progress_color = BLUE;
            } else {
                progress_color = GREEN;
            }

            for (int j = 0; j < 16; j++) {
                if (j < progress) {
                    printf("%s●%s", progress_color, RESET);
                } else {
                    printf("○");
                }
            }
            printf(" %s(%d/%d)%s\n", progress_color, tokens_generated[i], max_tokens[i], RESET);
        }
    }
    fflush(stdout);
}

long long generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
                   const char** input_seqs, int** output_tokens_batch, int batch_size, int steps) {
    OssSampler* sampler_oss = (OssSampler*)sampler;
    long long total_tokens_out = 0;

    if (!t_group || !t_d) {
        fprintf(stderr, "Thread-local expert parallel context not initialized\n");
        exit(EXIT_FAILURE);
    }

    // Encode all prompts in the batch
    int** prompt_tokens = (int**)malloc(batch_size * sizeof(int*));
    int* num_prompt_tokens = (int*)malloc(batch_size * sizeof(int));
    int max_prompt_len = 0;

    for (int b = 0; b < batch_size; b++) {
        const char* input_seq = input_seqs[b] ? input_seqs[b] : "";
        prompt_tokens[b] = (int*)malloc((strlen(input_seq) + 3) * sizeof(int));
        encode(tokenizer, input_seq, -1, -1, prompt_tokens[b], &num_prompt_tokens[b],
               t_d->config.initial_context_length);
        if (num_prompt_tokens[b] < 1) {
            fprintf(stderr, "Error: expected at least 1 prompt token for batch element %d\n", b);
            exit(EXIT_FAILURE);
        }
        max_prompt_len =
            (num_prompt_tokens[b] > max_prompt_len) ? num_prompt_tokens[b] : max_prompt_len;
    }

    // Batch processing variables
    int* current_tokens = (int*)malloc(batch_size * sizeof(int)); // Current token for each sequence
    int* pos = (int*)malloc(batch_size * sizeof(int));            // Position for each sequence
    bool* finished = (bool*)malloc(batch_size * sizeof(bool)); // Whether each sequence is finished
    int active_sequences = batch_size;

    // Initialize batch state
    for (int b = 0; b < batch_size; b++) {
        current_tokens[b] = prompt_tokens[b][0];
        pos[b] = 0;
        finished[b] = false;
    }

    for (int b = 0; b < batch_size; b++) {
        printf("#%d: ", b + 1);
        const char* first_piece = decode_piece(tokenizer, 200006, current_tokens[b]);
        safe_printf(first_piece);
        printf(" | ");
    }
    printf("\n");
    fflush(stdout);

    // Main generation loop
    int* tokens_generated = (int*)calloc(batch_size, sizeof(int));
    int* max_generation_tokens = (int*)malloc(batch_size * sizeof(int));

    for (int b = 0; b < batch_size; b++) {
        max_generation_tokens[b] = steps - 1 - num_prompt_tokens[b];
        if (max_generation_tokens[b] < 0)
            max_generation_tokens[b] = 0;
    }
    progress_bar(batch_size, tokens_generated, max_generation_tokens, finished);

    double total_forward_time = 0.0;
    int forward_calls = 0;

    while (active_sequences > 0) {
        int* batch_tokens = (int*)malloc(batch_size * sizeof(int));
        int* batch_positions = (int*)malloc(batch_size * sizeof(int));
        int* batch_indices = (int*)malloc(batch_size * sizeof(int));
        int valid_batch_size = 0;

        // Continuous batching
        for (int b = 0; b < batch_size && valid_batch_size < batch_size; b++) {
            if (!finished[b] && pos[b] + 1 < steps) {
                batch_tokens[valid_batch_size] = current_tokens[b];
                batch_positions[valid_batch_size] = pos[b];
                batch_indices[valid_batch_size] = b;
                valid_batch_size++;
            }
        }

        if (valid_batch_size == 0) {
            free(batch_tokens);
            free(batch_positions);
            free(batch_indices);
            break;
        }

        // Forward pass for the batch with mixed positions
        float* batch_logits = forward(t_d, t_group, batch_tokens, batch_positions, valid_batch_size,
                                      batch_indices, t_d->config.batch_size);

        // Process results for each sequence in the batch
        for (int i = 0; i < valid_batch_size; i++) {
            int b = batch_indices[i];
            int next_token;

            // Advance position
            pos[b]++;

            if (pos[b] < num_prompt_tokens[b]) {
                // Still in prompt phase - force next prompt token
                next_token = prompt_tokens[b][pos[b]];
            } else {
                // Generation phase - sample from logits
                float* seq_logits = batch_logits + i * t_d->config.vocab_size;
                next_token = sample_oss_gpu(sampler_oss, seq_logits);

                // Save output token
                int output_idx = pos[b] - num_prompt_tokens[b];
                if (output_tokens_batch[b]) {
                    output_tokens_batch[b][output_idx] = next_token;
                }
                total_tokens_out++;
                tokens_generated[b]++;
            }

            // Check for termination
            if (next_token == 199999 || next_token == 200002 || pos[b] + 1 >= steps) {
                finished[b] = true;
                active_sequences--;
                if (output_tokens_batch[b]) {
                    int output_idx = pos[b] - num_prompt_tokens[b] + 1;
                    output_tokens_batch[b][output_idx] = -1; // End marker
                }
            }

            current_tokens[b] = next_token;
        }

        clear_lines(batch_size);
        progress_bar(batch_size, tokens_generated, max_generation_tokens, finished);

        free(batch_tokens);
        free(batch_positions);
        free(batch_indices);
    }

    for (int b = 0; b < batch_size; b++) {
        free(prompt_tokens[b]);
    }
    free(prompt_tokens);
    free(num_prompt_tokens);
    free(current_tokens);
    free(pos);
    free(finished);
    free(tokens_generated);
    free(max_generation_tokens);

    return total_tokens_out;
}

long long inference(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
                    Requests* requests) {
    const int N = requests->num_reqs;
    long long total_tokens = 0;

#pragma omp parallel reduction(+ : total_tokens)
    {
        const int dp_rank = omp_get_thread_num();
        if (dp_rank >= g_dp_world_size) {
            fprintf(stderr, "OMP thread %d exceeds dp_world_size %d\n", dp_rank, g_dp_world_size);
            exit(EXIT_FAILURE);
        }

        OssExpertParallelGroup* group = &g_dp_groups[dp_rank];
        t_group = group;
        t_d = g_all_models[dp_rank];

        CHECK_HIP(hipSetDevice(dp_rank));

        // Per-thread sampler clone
        OssSampler* sampler_copy = (OssSampler*)malloc(sizeof(OssSampler));
        memcpy(sampler_copy, sampler, sizeof(OssSampler));

        // Build the list of request indices owned by this DP device
        int my_count = 0;
        for (int i = dp_rank; i < N; i += g_dp_world_size)
            my_count++;
        int* my_idx = (int*)malloc(sizeof(int) * (my_count > 0 ? my_count : 1));
        int w = 0;
        for (int i = dp_rank; i < N; i += g_dp_world_size)
            my_idx[w++] = i;

        const int batch_size = t_d->config.batch_size;

        // Batched on this replica
#pragma omp critical
        {
            printf("🚀 [DP device %d] Batched inference with batch_size = %d (%d reqs)\n", dp_rank,
                   batch_size, my_count);
            fflush(stdout);
        }

        for (int start = 0; start < my_count; start += batch_size) {
            const int cur_bs = ((my_count - start) < batch_size) ? (my_count - start) : batch_size;

            // Allocate batch views
            const char** input_seqs = (const char**)malloc(sizeof(char*) * cur_bs);
            int** out_tok_ptrs = (int**)malloc(sizeof(int*) * cur_bs);

            for (int i = 0; i < cur_bs; ++i) {
                const int req_idx = my_idx[start + i];
                input_seqs[i] = get_str_req_ptr(requests, req_idx);
                out_tok_ptrs[i] = get_tok_gen_ptr(requests, req_idx);
            }

#pragma omp critical
            {
                printf("[DP device %d] 📦 Batch %d/%d (req #%d → #%d)\n", dp_rank,
                       (start / batch_size) + 1, (my_count + batch_size - 1) / batch_size,
                       my_idx[start] + 1, my_idx[start + cur_bs - 1] + 1);
                fflush(stdout);
            }

            total_tokens += generate(transformer, tokenizer, (Sampler*)sampler_copy, input_seqs,
                                     out_tok_ptrs, cur_bs, requests->max_seq_len);

            free(input_seqs);
            free(out_tok_ptrs);
        }

        free(my_idx);
        free(sampler_copy);
    } // omp parallel

    return total_tokens;
}

#endif // GETP_RUN
