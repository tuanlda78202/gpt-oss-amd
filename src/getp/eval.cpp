#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#ifndef GETP_EVAL
#define GETP_EVAL

// ! -----------------------------Request Management---------------------------------------
typedef struct {
    int num_reqs;      // number of requests;
    int max_token_len; // maximum size of token
    int max_seq_len;   // maximum number of sequence
    char* str_reqs;    // buffer for requested strings
    int* tok_gens;     // buffer for generated tokens
} Requests;

void build_requests(Requests* reqs, int num_reqs, int max_token_len, int max_seq_len) {
    reqs->num_reqs = num_reqs;
    reqs->max_token_len = max_token_len;
    reqs->max_seq_len = max_seq_len;
    reqs->str_reqs = (char*)calloc(num_reqs * max_token_len * (max_seq_len + 1), sizeof(char));
    reqs->tok_gens = (int*)calloc(num_reqs * (max_seq_len + 1), sizeof(int));
    printf("requests size = %lu B\n",
           ((num_reqs * max_token_len * (max_seq_len + 1) * sizeof(char)) * 2));
    fflush(stdout);
}

void free_requests(Requests* reqs) {
    free(reqs->str_reqs);
    free(reqs->tok_gens);
}

char* get_str_req_ptr(Requests* reqs, int idx) {
    return reqs->str_reqs + idx * reqs->max_token_len * (reqs->max_seq_len + 1);
}

int* get_tok_gen_ptr(Requests* reqs, int idx) {
    return reqs->tok_gens + idx * (reqs->max_seq_len + 1);
}

/**
 * @brief Reads requests from a text file and populates a Requests buffer.
 *
 * Reads the first line of the file as the number of requests, optionally truncates
 * that count to |truncate_lines|, allocates request and token buffers via
 * build_requests, and copies up to that many subsequent lines into the request
 * string buffers.
 *
 * @param input_filename Path to the input text file. The file's first line must be an integer count.
 * @param max_token_len Maximum token length allocated per request string.
 * @param max_seq_len Maximum sequence length used for token generation buffers.
 * @param reqs Pointer to a Requests struct that will be initialized and populated.
 * @param truncate_lines If >0 and less than the file's declared count, limits the number of requests read.
 * @return int 0 on success.
 *
 * @note If the input file cannot be opened, the function prints an error message and exits the process with failure.
 */
int read_inputfile(const char* input_filename, int max_token_len, int max_seq_len, Requests* reqs,
                   int truncate_lines) {
    std::string filename = input_filename;
    int num_reqs = 0;

    std::ifstream openFile(filename.c_str());
    if (openFile.is_open()) {
        std::string line;

        std::getline(openFile, line);
        num_reqs = atoi(line.c_str());

        if (truncate_lines > 0 && truncate_lines < num_reqs) {
            printf("truncating input from %d to %d lines\n", num_reqs, truncate_lines);
            num_reqs = truncate_lines;
        }

        build_requests(reqs, num_reqs, max_token_len, max_seq_len);

        int idx = 0;
        while (std::getline(openFile, line)) {
            memcpy(get_str_req_ptr(reqs, idx), line.c_str(), line.size());
            idx++;
            if (idx >= num_reqs)
                break;
        }
        openFile.close();
    } else {
        fprintf(stderr, "cannot open the file: %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    printf("num requests: %d\n", reqs->num_reqs);

    return 0;
}

/**
 * @brief Writes generated token sequences for each request to a file, one sequence per line.
 *
 * Each request's generated tokens are written as space-separated integers on a single line.
 * For each request the sequence is written up to (but not including) the first negative value,
 * which acts as a sentinel terminating that sequence.
 *
 * @param output_filename Path to the file to create or overwrite.
 * @param reqs Pointer to a Requests structure containing generated token buffers and count.
 * @return int `0` on success.
 *
 * @note If the file cannot be opened for writing, the function prints an error message and exits the process with a failure status.
 */
int write_outputfile(const char* output_filename, Requests* reqs) {
    std::string filename = output_filename;

    std::ofstream writeFile(filename.c_str());
    if (writeFile.is_open()) {
        for (int i = 0; i < reqs->num_reqs; i++) {
            int* output_tokens = get_tok_gen_ptr(reqs, i);
            for (int j = 0; output_tokens[j] >= 0; j++) {
                if (j > 0)
                    writeFile << ' ';
                writeFile << output_tokens[j];
            }
            writeFile << '\n';
        }
        writeFile.close();
    } else {
        fprintf(stderr, "cannot write the file: %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    return 0;
}

// ! -----------------------------Eval Functions---------------------------------------
void warm_up(Transformer* transformer, Tokenizer* tokenizer, int batch_size, int use_kv16,
             int odd_window);
void finish(Transformer* transformer, Tokenizer* tokenizer);
long long inference(Transformer* transformer, Tokenizer* tokenizer, Sampler* sample,
                    Requests* requests);

/**
 * @brief Verifies that a generated output file matches a ground-truth file token-for-token.
 *
 * Compares the two files line-by-line, parsing each line into integer tokens, reports per-line
 * mismatches and a summary, and returns a status code indicating success, failure, or I/O error.
 *
 * @param generated_filename Path to the generated output file to verify.
 * @param ground_truth_filename Path to the ground-truth file to compare against.
 * @return int `0` if all requests match exactly; `1` if one or more requests have mismatches;
 * `-1` if either file cannot be opened.
 */
int verify_output(const char* generated_filename, const char* ground_truth_filename) {
    printf("\033[1;92m==================================================================\033["
           "0m\n\033[1;92m🔍 VERIFYING "
           "OUTPUT...\033[0m\n");
    fflush(stdout);

    std::ifstream generated_file(generated_filename);
    std::ifstream gt_file(ground_truth_filename);

    if (!generated_file.is_open()) {
        fprintf(stderr, "❌ Cannot open output file: %s\n", generated_filename);
        return -1;
    }

    if (!gt_file.is_open()) {
        fprintf(stderr, "❌ Cannot open GT file: %s\n", ground_truth_filename);
        return -1;
    }

    std::string gen_line, gt_line;
    int line_num = 1;
    int total_mismatches = 0;
    int total_lines = 0;

    while (std::getline(gt_file, gt_line)) {
        total_lines++;

        if (!std::getline(generated_file, gen_line)) {
            fprintf(stderr, "❌ Output has fewer lines than GT at line %d\n", line_num);
            total_mismatches++;
            break;
        }

        if (gt_line.empty() && gen_line.empty()) {
            line_num++;
            continue;
        }

        std::vector<int> gt_tokens, gen_tokens;

        std::istringstream gt_stream(gt_line);
        int token;
        while (gt_stream >> token) {
            gt_tokens.push_back(token);
        }

        std::istringstream gen_stream(gen_line);
        while (gen_stream >> token) {
            gen_tokens.push_back(token);
        }

        bool line_matches = true;
        if (gt_tokens.size() != gen_tokens.size()) {
            printf("⚠️ Request #%d: Length mismatch (GT: %zu tokens, Generated: %zu tokens)\n",
                   line_num, gt_tokens.size(), gen_tokens.size());
            line_matches = false;
        } else {
            for (size_t i = 0; i < gt_tokens.size(); i++) {
                if (gt_tokens[i] != gen_tokens[i]) {
                    printf(
                        "❌ Request %d: Token mismatch at position %zu (GT: %d, Generated: %d)\n",
                        line_num, i, gt_tokens[i], gen_tokens[i]);
                    line_matches = false;
                    break;
                }
            }
        }

        if (line_matches) {
            printf("✅ Request #%d: Match (%zu tokens)\n", line_num, gt_tokens.size());
        } else {
            total_mismatches++;
        }

        line_num++;
    }

    if (std::getline(generated_file, gen_line)) {
        fprintf(stderr, "❌ Output file has more lines than GT\n");
        total_mismatches++;
    }

    generated_file.close();
    gt_file.close();

    printf("\n📊 Verification Summary:\n");
    printf("Total requests checked: %d\n", total_lines);
    printf("Requests with mismatches: %d\n", total_mismatches);
    printf("Requests matching: %d\n", total_lines - total_mismatches);

    if (total_mismatches == 0) {
        printf("🎉 ✅ ALL TESTS PASSED!\n");
        return 0;
    } else {
        printf("❌ TESTS FAILED! %d requests have mismatches.\n", total_mismatches);
        return 1;
    }
}

/**
 * @brief Orchestrates end-to-end model evaluation: reads inputs, warms up the model, runs inference,
 * writes generated outputs, performs optional verification against a ground-truth file, and finalizes resources.
 *
 * Reads requests from input_filename, allocates request buffers, runs a warm-up routine and inference
 * (measuring and printing timings and throughput), writes generated tokens to output_filename, verifies
 * the output against verify_filename (or a default ground-truth file if NULL), and calls finish().
 * Exits the process on fatal I/O errors.
 *
 * @param input_filename Path to the input file containing requests.
 * @param output_filename Path where generated outputs will be written.
 * @param steps Maximum sequence length to use for tokenization and generation; if 0 or larger than the transformer's configured sequence length, the transformer's sequence length is used.
 * @param batch_size Number of requests processed per batch during warm-up and inference.
 * @param verify_filename Optional path to a ground-truth file for verification; if NULL a default ground-truth file is used.
 * @param truncate_lines If greater than 0 and less than the number of requests in the input file, limits processing to this many lines.
 * @param use_kv16 Configuration flag forwarded to warm_up indicating whether to use 16-key/value optimization.
 * @param odd_window Configuration parameter forwarded to warm_up that controls an odd-window behavior.
 */
void getp(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* input_filename,
          char* output_filename, int steps, int batch_size, char* verify_filename,
          int truncate_lines, int use_kv16, int odd_window) {
    // ! I/O
    Requests requests;
    int num_reqs;
    if (steps == 0 || steps > transformer->config.seq_len)
        steps = transformer->config.seq_len;
    if (input_filename == NULL || output_filename == NULL) {
        exit(EXIT_FAILURE);
    }
    if (EXIT_FAILURE == read_inputfile(input_filename, tokenizer->max_token_length, steps,
                                       &requests, truncate_lines)) {
        fprintf(stderr, "cannot read input file: %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    long start, end;

    // ! Warm up
    start = time_in_ms();
    printf("\033[1;91m==================================================================\033["
           "0m\n\033[1;91m🔥 WARMING UP...\033[0m");
    fflush(stdout);
    warm_up(transformer, tokenizer, batch_size, use_kv16, odd_window);
    end = time_in_ms();
    printf("⌛️ Warm up (s): %f\n", (double)(end - start) / 1000);
    fflush(stdout);

    // ! Inference
    start = time_in_ms();
    printf("\033[1;93m==================================================================\033["
           "0m\n\033[1;93m⚡️ RUNNING INFERENCE...\033[0m\n");
    fflush(stdout);
    long long num_gen_tokens = inference(transformer, tokenizer, sampler, &requests);
    end = time_in_ms();
    fprintf(stdout,
            "\n┌─────────────────────────────┐\n│ \033[1m⌛️ Time: %-13.6f\033[0m      │\n│ "
            "\033[1m⚡️ TPS: %-13.6f\033[0m       │\n└─────────────────────────────┘\n",
            (double)(end - start) / 1000, (num_gen_tokens) / (double)(end - start) * 1000);
    fflush(stdout);

    if (EXIT_FAILURE == write_outputfile(output_filename, &requests)) {
        fprintf(stderr, "cannot write output file: %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    // ! Verification
    const char* ground_truth_file = verify_filename ? verify_filename : "tests/gt/output_20b.txt";
    int verification_result = verify_output(output_filename, ground_truth_file);

    // ! Finish
    start = time_in_ms();
    finish(transformer, tokenizer);
    end = time_in_ms();
    printf("⌛️ Finish (s): %f\n", (double)(end - start) / 1000);
    fflush(stdout);

    // ! Verification
    if (verification_result == 0) {
        printf("✅ Verification: PASSED\n");
    } else {
        printf("❌ Verification: FAILED\n");
    }

    // free_requests(&requests);
}

#endif // GETP_EVAL