// ! DO NOT MODIFY THIS FILE

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

int read_inputfile(const char* input_filename, int max_token_len, int max_seq_len, Requests* reqs) {
    std::string filename = input_filename;
    int num_reqs = 0;

    std::ifstream openFile(filename.c_str());
    if (openFile.is_open()) {
        std::string line;

        // Read the number of Requests
        std::getline(openFile, line);
        num_reqs = atoi(line.c_str());

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

    printf("Num requests: %d\n", reqs->num_reqs);

    return 0;
}

int write_outputfile(const char* output_filename, Requests* reqs) {
    std::string filename = output_filename;

    // write File
    std::ofstream writeFile(filename.c_str());
    if (writeFile.is_open()) {
        for (int i = 0; i < reqs->num_reqs; i++) {
            int* output_tokens = get_tok_gen_ptr(reqs, i);
            for (int j = 0; output_tokens[j] >= 0; j++) {
                if (j > 0)
                    writeFile << ' ';
                writeFile << output_tokens[j];
            }
            writeFile << "\n";
        }
        writeFile.close();
    } else {
        fprintf(stderr, "cannot write the file: %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    return 0;
}

// ! -----------------------------Eval Functions---------------------------------------
void warm_up(Transformer* transformer, Tokenizer* tokenizer);
void finish(Transformer* transformer, Tokenizer* tokenizer);
long long inference(Transformer* transformer, Tokenizer* tokenizer, Sampler* sample,
                    Requests* requests);

int verify_output(const char* generated_filename, const char* ground_truth_filename) {
    printf("------------------------------\n🔍 Verifying output...\n");
    fflush(stdout);

    std::ifstream generated_file(generated_filename);
    std::ifstream gt_file(ground_truth_filename);

    if (!generated_file.is_open()) {
        fprintf(stderr, "❌ Cannot open generated file: %s\n", generated_filename);
        return -1;
    }

    if (!gt_file.is_open()) {
        fprintf(stderr, "❌ Cannot open ground truth file: %s\n", ground_truth_filename);
        return -1;
    }

    std::string gen_line, gt_line;
    int line_num = 1;
    int total_mismatches = 0;
    int total_lines = 0;

    while (std::getline(gt_file, gt_line)) {
        total_lines++;

        if (!std::getline(generated_file, gen_line)) {
            fprintf(stderr, "❌ Generated file has fewer lines than ground truth at line %d\n",
                    line_num);
            total_mismatches++;
            break;
        }

        // Skip empty lines
        if (gt_line.empty() && gen_line.empty()) {
            line_num++;
            continue;
        }

        // Parse tokens from both lines
        std::vector<int> gt_tokens, gen_tokens;

        // Parse ground truth tokens
        std::istringstream gt_stream(gt_line);
        int token;
        while (gt_stream >> token) {
            gt_tokens.push_back(token);
        }

        // Parse generated tokens
        std::istringstream gen_stream(gen_line);
        while (gen_stream >> token) {
            gen_tokens.push_back(token);
        }

        // Compare token sequences
        bool line_matches = true;
        if (gt_tokens.size() != gen_tokens.size()) {
            printf("❌ Request %d: Length mismatch (GT: %zu tokens, Generated: %zu tokens)\n",
                   line_num, gt_tokens.size(), gen_tokens.size());
            line_matches = false;
        } else {
            for (size_t i = 0; i < gt_tokens.size(); i++) {
                if (gt_tokens[i] != gen_tokens[i]) {
                    printf(
                        "❌ Request %d: Token mismatch at position %zu (GT: %d, Generated: %d)\n",
                        line_num, i, gt_tokens[i], gen_tokens[i]);
                    line_matches = false;
                    break; // Show only first mismatch per line
                }
            }
        }

        if (line_matches) {
            printf("✅ Request %d: Match (%zu tokens)\n", line_num, gt_tokens.size());
        } else {
            total_mismatches++;
        }

        line_num++;
    }

    // Check if generated file has extra lines
    if (std::getline(generated_file, gen_line)) {
        fprintf(stderr, "❌ Generated file has more lines than ground truth\n");
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

void getp(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* input_filename,
          char* output_filename, int steps) {
    // ! I/O
    Requests requests;
    int num_reqs;
    if (steps == 0 || steps > transformer->config.seq_len)
        steps = transformer->config.seq_len;
    if (input_filename == NULL || output_filename == NULL) {
        exit(EXIT_FAILURE);
    }
    if (EXIT_FAILURE ==
        read_inputfile(input_filename, tokenizer->max_token_length, steps, &requests)) {
        fprintf(stderr, "cannot read input file: %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    long start, end;

    // ! Warm up
    start = time_in_ms();
    printf("------------------------------\n🔥 Warming up...");
    fflush(stdout);
    warm_up(transformer, tokenizer);
    end = time_in_ms();
    printf("⌛️ Warm up (s): %f\n", (double)(end - start) / 1000);
    fflush(stdout);

    // ! Inference
    start = time_in_ms();
    printf("------------------------------\n⚡️ Running inference...\n\n");
    fflush(stdout);
    long long num_gen_tokens = inference(transformer, tokenizer, sampler, &requests);
    end = time_in_ms();
    // Your goal is to achieve best throughput(=reduce elapsed time)!
    fprintf(stdout, "\n️⌛️ Inference (s): %f, achieved throughput TPS (tok/s): %f\n",
            (double)(end - start) / 1000, (num_gen_tokens) / (double)(end - start) * 1000);
    fflush(stdout);

    if (EXIT_FAILURE == write_outputfile(output_filename, &requests)) {
        fprintf(stderr, "cannot write output file: %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    // ! TODO: remember remove verification later
    const char* ground_truth_file = "tests/gt/output.txt";
    int verification_result = verify_output(output_filename, ground_truth_file);

    // ! Finish
    start = time_in_ms();
    finish(transformer, tokenizer);
    end = time_in_ms();
    printf("\n⌛️ Finish (s): %f\n", (double)(end - start) / 1000);
    fflush(stdout);

    if (verification_result != 0) {
        printf("\n⚠️  Verification failed!\n");
    }

    // free_requests(&requests);
}

#endif // GETP_EVAL
