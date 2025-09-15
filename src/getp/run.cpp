#define GETP_SKIP_TYPEDEFS
#include "../../include/tokenizer.hpp"
#include "../forward.cpp"
#include "../model.cpp"
#include "../sampler.cpp"
#include "eval.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <stdbool.h>

#ifndef GETP_RUN
#define GETP_RUN

thread_local int dev_id = 0;
thread_local OssTransformerHybrid* t_d = nullptr;

// #define DP 1
// #define PP 1
int DP = atoi(getenv("DP"));
int PP = atoi(getenv("PP"));

static int g_num_devices = 1;
static int g_pp_stages = PP;

static int g_dp_groups = 1;         // how many independent copies (DP world)
static int g_devices_per_group = 1; // = g_pp_stages

static OssTransformerHybrid** g_models = nullptr;

void warm_up(Transformer* transformer, Tokenizer* tokenizer, int batch_size) {
    OssTransformer* transformer_oss = (OssTransformer*)transformer;
    transformer_oss->config.batch_size = batch_size;

    omp_set_max_active_levels(2);

    // Discover GPUs
    CHECK_HIP(hipGetDeviceCount(&g_num_devices));
    if (g_num_devices <= 0) {
        fprintf(stderr, "No HIP devices found.\n");
        exit(EXIT_FAILURE);
    }

    // Decide DP layout
    // If PP is enabled, we treat each PP island as a "device group".
    // Otherwise, each device is one DP group.
    if (g_pp_stages <= 1) {
        g_devices_per_group = 1;
        g_dp_groups = DP;
        if (g_dp_groups > g_num_devices)
            g_dp_groups = g_num_devices;
    } else {
        if (g_num_devices % g_pp_stages != 0) {
            fprintf(stderr, "GPU count (%d) is not divisible by PP (%d).\n", g_num_devices,
                    g_pp_stages);
            exit(EXIT_FAILURE);
        }
        g_devices_per_group = g_pp_stages;
        g_dp_groups = DP;
        if (g_dp_groups > (g_num_devices / g_pp_stages))
            g_dp_groups = (g_num_devices / g_pp_stages);
    }

    printf("\n[DP] devices=%d, pp_stages=%d, dp_groups=%d, devices_per_group=%d\n", g_num_devices,
           g_pp_stages, g_dp_groups, g_devices_per_group);
    omp_set_num_threads(g_dp_groups);

    // Allocate one model replica per DP group (each replica will select its
    // first device as "home" for warmup copy; PP code, if used, will fan-out).
    g_models = (OssTransformerHybrid**)malloc(sizeof(OssTransformerHybrid*) * g_dp_groups);
    if (!g_models) {
        fprintf(stderr, "malloc g_models failed\n");
        exit(EXIT_FAILURE);
    }

    // Build/transfer one replica per DP group
    // Note: we pick a representative device for each group to call
    // copy_transformer_to_device_hybrid. If you have internal PP that spreads within the group,
    // that code will take over later.
#pragma omp parallel for num_threads(DP* PP) schedule(static)
    for (int g = 0; g < g_dp_groups; ++g) {
        int device_base = g * g_devices_per_group; // first device of this DP group
        CHECK_HIP(hipSetDevice(device_base));

        g_models[g] = (OssTransformerHybrid*)malloc(sizeof(OssTransformerHybrid));
        if (!g_models[g]) {
            fprintf(stderr, "malloc model for group %d failed\n", g);
            exit(EXIT_FAILURE);
        }

        // Use the new grouped loader so PP shards land on device_base..device_base+PP-1
        copy_transformer_to_device_hybrid_grouped(transformer_oss, g_models[g], device_base,
                                                  g_devices_per_group);

        size_t free_mem, total_mem;
        CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
#pragma omp critical
        {
            printf("\n--- HYBRID WARM-UP COMPLETE (group %d on device %d) ---\n", g, device_base);
            printf("GPU Memory Status: Total %.2f GB, Used %.2f GB, Free %.2f GB\n",
                   total_mem / (1024.0 * 1024.0 * 1024.0),
                   (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0),
                   free_mem / (1024.0 * 1024.0 * 1024.0));
            printf("-----------------------------------------------\n");
        }
    }
}

void finish(Transformer* transformer, Tokenizer* tokenizer) {
    // print_batch_timing_summary();
    if (g_models) {
#pragma omp parallel for schedule(static) num_threads(g_dp_groups)
        for (int g = 0; g < g_dp_groups; ++g) {
            int home_device = g * g_devices_per_group;
            CHECK_HIP(hipSetDevice(home_device));
            if (g_models[g]) {
                free_transformer_on_device_hybrid(g_models[g]);
                free(g_models[g]);
            }
        }
        free(g_models);
        g_models = nullptr;
    }

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

        float* batch_logits = forward(t_d, batch_tokens, batch_positions, valid_batch_size,
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
        const int tid = omp_get_thread_num();
        const int group = tid; // one OMP thread per DP group
        const int home = group * g_devices_per_group;

        // Bind to group's home device and replica
        CHECK_HIP(hipSetDevice(home));
        dev_id = group;
        t_d = g_models[group];

        // Per-thread sampler clone
        OssSampler* sampler_copy = (OssSampler*)malloc(sizeof(OssSampler));
        memcpy(sampler_copy, sampler, sizeof(OssSampler));

        // Build the list of request indices owned by this DP group: i % g_dp_groups == group
        int my_count = 0;
        for (int i = group; i < N; i += g_dp_groups)
            my_count++;
        int* my_idx = (int*)malloc(sizeof(int) * (my_count > 0 ? my_count : 1));
        int w = 0;
        for (int i = group; i < N; i += g_dp_groups)
            my_idx[w++] = i;

        const int batch_size = t_d->config.batch_size;

        // Batched on this replica
#pragma omp critical
        {
            printf("🚀 [DP group %d] Batched inference with batch_size = %d (%d reqs)\n", group,
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
                printf("[DP group %d] 📦 Batch %d/%d (req #%d → #%d)\n", group,
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
