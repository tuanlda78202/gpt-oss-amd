#define GETP_SKIP_TYPEDEFS
#include "../../include/tokenizer.hpp"
#include "../forward.cpp"
#include "../model.cpp"
#include "../sampler.cpp"
#include "eval.cpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <stdbool.h>
#include <vector>

#ifndef GETP_RUN
#define GETP_RUN

static OssTransformerHybrid** g_models = nullptr; // store all the models
thread_local OssTransformerHybrid* t_d = nullptr; // local model

struct ThreadLocalBatchBuffers {
    int capacity = 0;
    int* tokens = nullptr;
    int* positions = nullptr;
    int* indices = nullptr;
    int* sample_results_d = nullptr;
    int* sample_results_h = nullptr;
    int* sample_owner = nullptr;
    int* sample_slots = nullptr;
};

struct ThreadLocalRequestBuffers {
    int capacity = 0;
    const char** input_ptrs = nullptr;
    int** output_ptrs = nullptr;
};

static thread_local ThreadLocalBatchBuffers g_batch_buffers;
static thread_local ThreadLocalRequestBuffers g_request_buffers;

static void ensure_batch_buffers(int required) {
    ThreadLocalBatchBuffers& buf = g_batch_buffers;
    if (required <= buf.capacity)
        return;

    auto release_host = [](int*& ptr) {
        if (ptr) {
            CHECK_HIP(hipHostFree(ptr));
            ptr = nullptr;
        }
    };

    if (buf.sample_results_d) {
        CHECK_HIP(hipFree(buf.sample_results_d));
        buf.sample_results_d = nullptr;
    }
    release_host(buf.tokens);
    release_host(buf.positions);
    release_host(buf.indices);
    release_host(buf.sample_results_h);

    if (buf.sample_owner) {
        free(buf.sample_owner);
        buf.sample_owner = nullptr;
    }
    if (buf.sample_slots) {
        free(buf.sample_slots);
        buf.sample_slots = nullptr;
    }

    CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&buf.tokens), required * sizeof(int)));
    CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&buf.positions), required * sizeof(int)));
    CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&buf.indices), required * sizeof(int)));
    CHECK_HIP(hipMalloc(&buf.sample_results_d, required * sizeof(int)));
    CHECK_HIP(
        hipHostMalloc(reinterpret_cast<void**>(&buf.sample_results_h), required * sizeof(int)));

    buf.sample_owner = reinterpret_cast<int*>(malloc(required * sizeof(int)));
    buf.sample_slots = reinterpret_cast<int*>(malloc(required * sizeof(int)));

    if (!buf.sample_owner || !buf.sample_slots) {
        fprintf(stderr, "Failed to allocate thread-local batch buffers\n");
        exit(EXIT_FAILURE);
    }

    buf.capacity = required;
}

static void ensure_request_buffers(int required) {
    ThreadLocalRequestBuffers& buf = g_request_buffers;
    if (required <= buf.capacity)
        return;

    if (buf.input_ptrs) {
        free(buf.input_ptrs);
        buf.input_ptrs = nullptr;
    }
    if (buf.output_ptrs) {
        free(buf.output_ptrs);
        buf.output_ptrs = nullptr;
    }

    buf.input_ptrs = reinterpret_cast<const char**>(
        malloc(static_cast<size_t>(required) * sizeof(*buf.input_ptrs)));
    buf.output_ptrs =
        reinterpret_cast<int**>(malloc(static_cast<size_t>(required) * sizeof(*buf.output_ptrs)));
    if (!buf.input_ptrs || !buf.output_ptrs) {
        fprintf(stderr, "Failed to allocate request buffers\n");
        exit(EXIT_FAILURE);
    }

    buf.capacity = required;
}

#define DP 8
static int num_gpus = 1;

void warm_up(Transformer* transformer, Tokenizer* tokenizer, int batch_size, int use_kv16) {
    OssTransformer* transformer_oss = (OssTransformer*)transformer;
    transformer_oss->config.batch_size = batch_size;
    transformer_oss->config.seq_len = 1024;

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

        copy_transformer_to_device(transformer_oss, g_models[g], use_kv16);

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
                free_transformer_on_device(g_models[g]);
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
                   const char** input_seqs, int** output_tokens_batch, int batch_size, int steps,
                   bool show_progress) {
    OssSampler* sampler_oss = (OssSampler*)sampler;
    const bool use_async_sampling = (sampler_oss->temperature == 0.0f);
    long long total_tokens_out = 0;

    ensure_batch_buffers(t_d->config.batch_size);
    ThreadLocalBatchBuffers& buffers = g_batch_buffers;
    int* batch_tokens = buffers.tokens;
    int* batch_positions = buffers.positions;
    int* batch_indices = buffers.indices;
    int* sample_owner = buffers.sample_owner;
    int* sample_slots = buffers.sample_slots;
    int* sample_results_h = buffers.sample_results_h;
    int* sample_results_d = buffers.sample_results_d;

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
        if (num_prompt_tokens[b] > max_prompt_len)
            max_prompt_len = num_prompt_tokens[b];
    }

    // Batch processing variables
    int* current_tokens = (int*)malloc(batch_size * sizeof(int));
    int* pos = (int*)malloc(batch_size * sizeof(int));
    bool* finished = (bool*)malloc(batch_size * sizeof(bool));
    int active_sequences = batch_size;

    for (int b = 0; b < batch_size; b++) {
        current_tokens[b] = prompt_tokens[b][0];
        pos[b] = 0;
        finished[b] = false;
    }

    if (show_progress) {
        for (int b = 0; b < batch_size; b++) {
            printf("#%d: ", b + 1);
            const char* first_piece = decode_piece(tokenizer, 200006, current_tokens[b]);
            safe_printf(first_piece);
            printf(" | ");
        }
        printf("\n");
        fflush(stdout);
    }

    int* tokens_generated = (int*)calloc(batch_size, sizeof(int));
    int* max_generation_tokens = (int*)malloc(batch_size * sizeof(int));

    for (int b = 0; b < batch_size; b++) {
        int remaining = steps - 1 - num_prompt_tokens[b];
        max_generation_tokens[b] = remaining > 0 ? remaining : 0;
    }

    const int progress_stride = 1;
    int iteration = 0;

    if (show_progress) {
        progress_bar(batch_size, tokens_generated, max_generation_tokens, finished);
    }

    auto commit_generation = [&](int b_idx, int next_token, bool counted) {
        if (counted && output_tokens_batch[b_idx]) {
            int output_idx = pos[b_idx] - num_prompt_tokens[b_idx];
            if (output_idx >= 0) {
                output_tokens_batch[b_idx][output_idx] = next_token;
            }
        }
        if (counted) {
            total_tokens_out++;
            tokens_generated[b_idx]++;
        }
        if ((next_token == 199999 || next_token == 200002 || pos[b_idx] + 1 >= steps) &&
            !finished[b_idx]) {
            finished[b_idx] = true;
            active_sequences--;
            if (output_tokens_batch[b_idx]) {
                int end_idx = pos[b_idx] - num_prompt_tokens[b_idx] + 1;
                if (end_idx >= 0) {
                    output_tokens_batch[b_idx][end_idx] = -1;
                }
            }
        }
        current_tokens[b_idx] = next_token;
    };

    while (active_sequences > 0) {
        int valid_batch_size = 0;
        for (int b = 0; b < batch_size && valid_batch_size < batch_size; b++) {
            if (!finished[b] && pos[b] + 1 < steps) {
                batch_tokens[valid_batch_size] = current_tokens[b];
                batch_positions[valid_batch_size] = pos[b];
                batch_indices[valid_batch_size] = b;
                valid_batch_size++;
            }
        }

        if (valid_batch_size == 0)
            break;

        float* batch_logits = forward(t_d, batch_tokens, batch_positions, valid_batch_size,
                                      batch_indices, t_d->config.batch_size);

        int pending_samples = 0;
        for (int i = 0; i < valid_batch_size; i++) {
            int b_idx = batch_indices[i];
            pos[b_idx]++;

            if (pos[b_idx] < num_prompt_tokens[b_idx]) {
                int next_token = prompt_tokens[b_idx][pos[b_idx]];
                commit_generation(b_idx, next_token, false);
            } else if (use_async_sampling) {
                sample_owner[pending_samples] = b_idx;
                sample_slots[pending_samples] = i;
                pending_samples++;
            } else {
                float* seq_logits = batch_logits + i * t_d->config.vocab_size;
                int next_token = sample_oss_gpu(sampler_oss, seq_logits);
                commit_generation(b_idx, next_token, true);
            }
        }

        if (use_async_sampling && pending_samples > 0) {
            for (int s = 0; s < pending_samples; ++s) {
                float* seq_logits = batch_logits + sample_slots[s] * t_d->config.vocab_size;
                sample_argmax(seq_logits, t_d->config.vocab_size, sample_results_d + s);
            }
            CHECK_HIP(hipMemcpyAsync(sample_results_h, sample_results_d,
                                     pending_samples * sizeof(int), hipMemcpyDeviceToHost, 0));
            CHECK_HIP(hipStreamSynchronize(0));
            for (int s = 0; s < pending_samples; ++s) {
                int b_idx = sample_owner[s];
                int next_token = sample_results_h[s];
                commit_generation(b_idx, next_token, true);
            }
        }

        iteration++;
        if (show_progress && ((iteration % progress_stride == 0) || active_sequences == 0)) {
            clear_lines(batch_size);
            progress_bar(batch_size, tokens_generated, max_generation_tokens, finished);
        }
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
    const int total_requests = requests->num_reqs;
    if (total_requests <= 0)
        return 0;

    const int max_batch_size = g_models[0]->config.batch_size;
    std::vector<int> batch_starts;
    batch_starts.reserve((total_requests + max_batch_size - 1) / max_batch_size);
    for (int start = 0; start < total_requests; start += max_batch_size)
        batch_starts.push_back(start);

    std::atomic<int> next_batch{0};
    long long total_tokens = 0;

#pragma omp parallel reduction(+ : total_tokens)
    {
        const int device = omp_get_thread_num();
        CHECK_HIP(hipSetDevice(device));
        t_d = g_models[device];

        OssSampler* sampler_copy = (OssSampler*)malloc(sizeof(OssSampler));
        memcpy(sampler_copy, sampler, sizeof(OssSampler));

        ensure_request_buffers(max_batch_size);
        ThreadLocalRequestBuffers& req_buf = g_request_buffers;

        const bool is_log_device = (device == 0);
        int local_batches = 0;

#pragma omp critical
        {
            printf("🚀 [DP device %d] Ready with batch_size = %d\n", device,
                   t_d->config.batch_size);
            fflush(stdout);
        }

        while (true) {
            int batch_id = next_batch.fetch_add(1, std::memory_order_relaxed);
            if (batch_id >= static_cast<int>(batch_starts.size()))
                break;

            const int start = batch_starts[batch_id];
            const int cur_bs = std::min(max_batch_size, total_requests - start);

            const char** input_seqs = req_buf.input_ptrs;
            int** out_tok_ptrs = req_buf.output_ptrs;

            for (int i = 0; i < cur_bs; ++i) {
                const int req_idx = start + i;
                input_seqs[i] = get_str_req_ptr(requests, req_idx);
                out_tok_ptrs[i] = get_tok_gen_ptr(requests, req_idx);
            }

            const bool enable_progress = is_log_device && ((local_batches % 8) == 0);
            total_tokens += generate(transformer, tokenizer, (Sampler*)sampler_copy, input_seqs,
                                     out_tok_ptrs, cur_bs, requests->max_seq_len, enable_progress);
            local_batches++;
        }

        free(sampler_copy);
    }

    return total_tokens;
}

#endif // GETP_RUN
