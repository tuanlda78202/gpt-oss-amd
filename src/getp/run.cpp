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
#include <stdbool.h>

#ifndef GETP_RUN
#define GETP_RUN

OssTransformerHybrid* t_d;

void warm_up(Transformer* transformer, Tokenizer* tokenizer) {
    OssTransformer* transformer_oss = (OssTransformer*)transformer;

    t_d = (OssTransformerHybrid*)malloc(sizeof(OssTransformerHybrid));

    // TODO: set first 16 rq not working correctly
    transformer_oss->config.batch_size = 4;

    copy_transformer_to_device_hybrid(transformer_oss, t_d);

    // ! GPU stats
    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    size_t used_mem = total_mem - free_mem;
    printf("\n--- HYBRID WARM-UP COMPLETE ---\n");
    printf("GPU Memory Status:\n");
    printf("  Total: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Used: %.2f GB\n", used_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Free: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
    printf("-------------------------------\n");
}

void finish(Transformer* transformer, Tokenizer* tokenizer) {
    free_transformer_on_device_hybrid(t_d);
    free(t_d);

    // ! GPU stats
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
           t_d->config.initial_context_length);
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
        float* logits = forward_hybrid(t_d, token, pos);

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

// Batched inference for multiple requests
long long batched_getp_generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
                                const char** input_seqs, int** output_tokens_batch, int batch_size,
                                int steps) {
    OssSampler* sampler_oss = (OssSampler*)sampler;
    long long total_tokens_out = 0;

    // Encode all prompts in the batch
    int* prompt_tokens[batch_size];
    int num_prompt_tokens[batch_size];
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
    int current_tokens[batch_size]; // Current token for each sequence
    int pos[batch_size];            // Position for each sequence
    bool finished[batch_size];      // Whether each sequence is finished
    int active_sequences = batch_size;

    // Initialize batch state
    for (int b = 0; b < batch_size; b++) {
        current_tokens[b] = prompt_tokens[b][0];
        pos[b] = 0;
        finished[b] = false;
    }

    // Print first tokens
    for (int b = 0; b < batch_size; b++) {
        printf("Batch %d: ", b);
        const char* first_piece = decode_piece(tokenizer, 200006, current_tokens[b]);
        safe_printf(first_piece);
        printf(" | ");
    }
    printf("\n");
    fflush(stdout);

    // Main generation loop
    while (active_sequences > 0) {
        bool process_batch = false;
        int batch_tokens[batch_size];
        int batch_positions[batch_size];
        int batch_indices[batch_size];
        int valid_batch_size = 0;

        // Collect sequences that need processing at the same position
        int target_pos = -1;
        for (int b = 0; b < batch_size; b++) {
            if (!finished[b] && pos[b] < steps) {
                if (target_pos == -1)
                    target_pos = pos[b];
                if (pos[b] == target_pos) {
                    batch_tokens[valid_batch_size] = current_tokens[b];
                    batch_positions[valid_batch_size] = pos[b];
                    batch_indices[valid_batch_size] = b;
                    valid_batch_size++;
                    process_batch = true;
                }
            }
        }

        if (!process_batch || valid_batch_size == 0)
            break;

        // Forward pass for the batch
        float* batch_logits = forward_hybrid_batch(t_d, batch_tokens, target_pos, valid_batch_size);

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
            }

            // Check for termination
            if (next_token == 199999 || next_token == 200002 || pos[b] >= steps) {
                finished[b] = true;
                active_sequences--;
                if (output_tokens_batch[b]) {
                    int output_idx = pos[b] - num_prompt_tokens[b] + 1;
                    output_tokens_batch[b][output_idx] = -1; // End marker
                }
                printf("\nBatch %d finished after %d tokens\n", b, pos[b]);
            } else {
                // Print generated token
                const char* piece = decode_piece(tokenizer, current_tokens[b], next_token);
                // TODO: make log better
                // printf("B%d: %s\n", b, piece);
                // fflush(stdout);
            }

            current_tokens[b] = next_token;
        }
    }

    // Cleanup
    for (int b = 0; b < batch_size; b++) {
        free(prompt_tokens[b]);
    }

    printf("\nBatched generation complete. Total tokens generated: %lld\n", total_tokens_out);
    return total_tokens_out;
}

long long inference(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
                    Requests* requests) {
    long long num_token_out = 0;
    int batch_size = t_d->config.batch_size;

    if (batch_size <= 1 || requests->num_reqs == 1) {
        printf("🛑 Using single-sequence inference\n");
        fflush(stdout);
        for (int idx = 0; idx < requests->num_reqs; ++idx) {
            const char* input_seq = get_str_req_ptr(requests, idx);
            int* output_tokens = get_tok_gen_ptr(requests, idx);
            num_token_out += simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                                                  output_tokens, requests->max_seq_len);
        }
    } else {
        printf("🛑 Using batched inference with batch_size = %d\n", batch_size);
        fflush(stdout);

        for (int start_idx = 0; start_idx < requests->num_reqs; start_idx += batch_size) {
            int current_batch_size = ((requests->num_reqs - start_idx) < batch_size)
                                         ? (requests->num_reqs - start_idx)
                                         : batch_size;

            // Prepare batch
            const char* input_seqs[batch_size];
            int* output_tokens_batch[batch_size];

            for (int i = 0; i < current_batch_size; i++) {
                int req_idx = start_idx + i;
                input_seqs[i] = get_str_req_ptr(requests, req_idx);
                output_tokens_batch[i] = get_tok_gen_ptr(requests, req_idx);
            }

            // Fill remaining slots with nulls if needed
            for (int i = current_batch_size; i < batch_size; i++) {
                input_seqs[i] = "";
                output_tokens_batch[i] = nullptr;
            }

            // Process batch
            num_token_out += batched_getp_generate(transformer, tokenizer, sampler, input_seqs,
                                                   output_tokens_batch, current_batch_size,
                                                   requests->max_seq_len);
        }
    }

    return num_token_out;
}

#endif // GETP_RUN
