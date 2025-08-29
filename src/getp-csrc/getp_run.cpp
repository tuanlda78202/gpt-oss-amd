// TODO: Modify this file to optimize end-to-end throughput
#define GETP_SKIP_TYPEDEFS
#include "../../include/tokenizer.hpp"
#include "../forward.cpp"
#include "../model.cpp"
#include "../sampler.cpp"
#include "getp_eval.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#ifndef GETP_RUN
#define GETP_RUN

OssTransformerHybrid* t_d;

void warm_up(Transformer* transformer, Tokenizer* tokenizer) {
    // Do not inference here
    // You should handle the warm-up process
    // TODO:
    // - Memory allocation
    // - Load model
    // - ...
    OssTransformer* transformer_oss = (OssTransformer*)transformer;

    t_d = (OssTransformerHybrid*)malloc(sizeof(OssTransformerHybrid));

    copy_transformer_to_device_hybrid(transformer_oss, t_d);

    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    size_t used_mem = total_mem - free_mem;
    printf("\n=== HYBRID WARM-UP COMPLETE ===\n");
    printf("GPU Memory Status:\n");
    printf("  Total: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Used: %.2f GB\n", used_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Free: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
    printf("================================\n");
}

void finish(Transformer* transformer, Tokenizer* tokenizer) {
    // Do not inference here
    // You should handle the finish process
    // TODO:
    // - Memory deallocation
    // - Unload model
    // - ...

    free_transformer_on_device_hybrid(t_d);
    free(t_d);

    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    size_t used_mem = total_mem - free_mem;
    printf("\n=== HYBRID FINISH COMPLETE ===\n");
    printf("GPU Memory Status:\n");
    printf("  Total: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Used: %.2f GB\n", used_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Free: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
    printf("================================\n");
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
    encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens,
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
    // const char* first_piece = decode_piece(tokenizer, 200006, token);
    // safe_printf(first_piece);
    // fflush(stdout);

    while (pos < steps) {
        // forward the transformer to get logits for the next token
        float* logits = forward_hybrid(t_d, token, pos);

        // advance the state machine
        pos++;
        if (pos < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt
            // token
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
    long long num_token_out = 0;

    for (int idx = 0; idx < requests->num_reqs; ++idx) {
        const char* input_seq = get_str_req_ptr(requests, idx);
        int* output_tokens = get_tok_gen_ptr(requests, idx);
        num_token_out += simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                                              output_tokens, requests->max_seq_len);
    }
    return num_token_out;
}

#endif // GETP_RUN
