// TODO: Modify this file to optimize end-to-end throughput
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

#ifndef GETP_RUN
#define GETP_RUN

thread_local int TLS_DP_GROUP = 0;
thread_local OssTransformerHybrid* TLS_TD = nullptr;

static int g_num_devices = 1;
static int g_pp_stages = std::getenv("PP") ? std::atoi(std::getenv("PP")) : 1;

static int g_dp_groups = 1;         // how many independent copies (DP world)
static int g_devices_per_group = 1; // = g_pp_stages

static OssTransformerHybrid** g_models = nullptr;

// Map a request index to a DP group id
static inline int dp_group_for_req(int req_index) {
    // simple round-robin across DP groups
    return (req_index % g_dp_groups);
}

static int getenv_int(const char* key, int fallback) {
    const char* v = getenv(key);
    if (!v)
        return fallback;
    char* endp = nullptr;
    long x = strtol(v, &endp, 10);
    if (!endp || *endp != '\0' || x <= 0)
        return fallback;
    return (int)x;
}

void warm_up(Transformer* transformer, Tokenizer* tokenizer) {
    OssTransformer* transformer_oss = (OssTransformer*)transformer;

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
        g_dp_groups = getenv_int("DP", g_num_devices); // default: use all GPUs
        if (g_dp_groups > g_num_devices)
            g_dp_groups = g_num_devices;
    } else {
        if (g_num_devices % g_pp_stages != 0) {
            fprintf(stderr, "GPU count (%d) is not divisible by PP (%d).\n", g_num_devices,
                    g_pp_stages);
            exit(EXIT_FAILURE);
        }
        g_devices_per_group = g_pp_stages;
        g_dp_groups = getenv_int("DP", g_num_devices / g_pp_stages);
        if (g_dp_groups > (g_num_devices / g_pp_stages))
            g_dp_groups = (g_num_devices / g_pp_stages);
    }

    printf("[DP] devices=%d, pp_stages=%d, dp_groups=%d, devices_per_group=%d\n", g_num_devices,
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
#pragma omp parallel for schedule(static)
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
    if (g_models) {
#pragma omp parallel for schedule(static)
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

long long simple_getp_generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
                               const char* input_seq, int* output_tokens, int steps) {
    // <|start|>: 200006
    // <|end|>: 200007
    // <|return|>: 200002
    // <|message|>: 200008
    // <|channel|>: 200005
    // <|constrain|>: 200003
    // <|endoftext|>: 199999

    // Inference here
    OssSampler* sampler_oss = (OssSampler*)sampler;

    const char* empty_prompt = "";
    if (input_seq == NULL) {
        input_seq = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens =
        (int*)malloc((strlen(input_seq) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
           TLS_TD->config.initial_context_length);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence

    // print the very first token should be removed
    const char* first_piece = decode_piece(tokenizer, 200006, token);
    safe_printf(first_piece);
    fflush(stdout);

    while (pos < steps) {
        // forward the transformer to get logits for the next token
        extern thread_local int TLS_DP_GROUP;
        float* logits = forward_hybrid(g_models[TLS_DP_GROUP], token, pos);

        // advance the state machine
        pos++;
        if (pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos];
        } else {
            // otherwise sample the next token from the logits
            next = sample_oss_gpu(sampler_oss, logits);
            // save the output token, it will be printed to file
            output_tokens[pos - num_prompt_tokens] = next;
        }

        // data-dependent terminating condition: the EOS (=199999 or =200002) token
        // delimits sequences
        if (next == 199999 || next == 200002) {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        // should be removed
        const char* piece = decode_piece(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);

        token = next;
    }

    // should be removed
    printf("\n");

    // Marker for end of sequence
    output_tokens[pos - num_prompt_tokens + 1] = -1;

    free(prompt_tokens);

    return pos - num_prompt_tokens + 1;
}

long long inference(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
                    Requests* requests) {
    const int N = requests->num_reqs;
    long long total_tokens = 0;

#pragma omp parallel reduction(+ : total_tokens)
    {
        const int tid = omp_get_thread_num();
        const int group = tid; // one thread per DP group
        const int home = group * g_devices_per_group;
        CHECK_HIP(hipSetDevice(home));

        // thread-local sampler copy
        OssSampler* sampler_copy = (OssSampler*)malloc(sizeof(OssSampler));
        memcpy(sampler_copy, sampler, sizeof(OssSampler));

        // bind this thread to its replica
        TLS_DP_GROUP = group;
        TLS_TD = g_models[group];

        // consume only my group's requests
        for (int idx = group; idx < N; idx += g_dp_groups) {
            const char* input_seq = get_str_req_ptr(requests, idx);
            int* out_tokens = get_tok_gen_ptr(requests, idx);

            total_tokens += simple_getp_generate(transformer, tokenizer, (Sampler*)sampler_copy,
                                                 input_seq, out_tokens, requests->max_seq_len);
        }
        free(sampler_copy);
    }
    return total_tokens;
}

#endif // GETP_RUN
