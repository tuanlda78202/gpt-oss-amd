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

static OssTransformerHybrid** g_models = nullptr; // store all the models
thread_local OssTransformerHybrid* t_d = nullptr; // local model

#define DP 8
static int num_gpus = 1;

void warm_up(Transformer* transformer, Tokenizer* tokenizer, int batch_size) {
    OssTransformer* transformer_oss = (OssTransformer*)transformer;
    transformer_oss->config.batch_size = batch_size;

    // Discover GPUs
    CHECK_HIP(hipGetDeviceCount(&num_gpus));
    if (num_gpus <= 0) {
        fprintf(stderr, "No HIP devices found.\n");
        exit(EXIT_FAILURE);
    }

    num_gpus = std::max(1, std::min(num_gpus, DP));
    printf("\n[DP] devices=%d, batch_size=%d\n", num_gpus, batch_size);
    omp_set_num_threads(num_gpus);

    g_models = (OssTransformerHybrid**)malloc(sizeof(OssTransformerHybrid*) * num_gpus);
    if (!g_models) {
        fprintf(stderr, "malloc g_models failed\n");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for num_threads(num_gpus) schedule(static)
    for (int g = 0; g < num_gpus; ++g) {
        CHECK_HIP(hipSetDevice(g));

        g_models[g] = (OssTransformerHybrid*)malloc(sizeof(OssTransformerHybrid));
        if (!g_models[g]) {
            fprintf(stderr, "malloc model for device %d failed\n", g);
            exit(EXIT_FAILURE);
        }

        copy_transformer_to_device_hybrid(transformer_oss, g_models[g]);

        size_t free_mem, total_mem;
        CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
#pragma omp critical
        {
            printf("\n--- HYBRID WARM-UP COMPLETE (device %d) ---\n", g);
            printf("GPU Memory Status: Total %.2f GB, Used %.2f GB, Free %.2f GB\n",
                   total_mem / (1024.0 * 1024.0 * 1024.0),
                   (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0),
                   free_mem / (1024.0 * 1024.0 * 1024.0));
            printf("-----------------------------------------------\n");
        }
    }

    reset_batch_timings();
}

void finish(Transformer* transformer, Tokenizer* tokenizer) {
    print_batch_timing_summary();

    if (g_models) {
#pragma omp parallel for num_threads(num_gpus) schedule(static)
        for (int g = 0; g < num_gpus; ++g) {
            CHECK_HIP(hipSetDevice(g));
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
        const int device = tid; // one OMP thread per device

        CHECK_HIP(hipSetDevice(device));
        t_d = g_models[device];

        // Per-thread sampler clone
        OssSampler* sampler_copy = (OssSampler*)malloc(sizeof(OssSampler));
        memcpy(sampler_copy, sampler, sizeof(OssSampler));

        // Build the list of request indices owned by this DP device
        int my_count = 0;
        for (int i = device; i < N; i += num_gpus)
            my_count++;
        int* my_idx = (int*)malloc(sizeof(int) * (my_count > 0 ? my_count : 1));
        int w = 0;
        for (int i = device; i < N; i += num_gpus)
            my_idx[w++] = i;

        const int batch_size = t_d->config.batch_size;

        // Batched on this replica
#pragma omp critical
        {
            printf("🚀 [DP device %d] Batched inference with batch_size = %d (%d reqs)\n", device,
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
                printf("[DP device %d] 📦 Batch %d/%d (req #%d → #%d)\n", device,
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
