// DO NOT MODIFY THIS FILE

#include <fstream>
#include <iostream>

#ifndef GETP_EVAL
#define GETP_EVAL

typedef struct {
  int num_reqs;       // number of requests;
  int max_token_len;  // maximum size of token
  int max_seq_len;    // maximum number of sequence
  char* str_reqs;     // buffer for requested strings
  int* tok_gens;      // buffer for generated tokens
} Requests;

void build_requests(Requests* reqs, int num_reqs, int max_token_len, int max_seq_len) {
  reqs->num_reqs = num_reqs;
  reqs->max_token_len = max_token_len;
  reqs->max_seq_len = max_seq_len;
  reqs->str_reqs = (char*)calloc(num_reqs * max_token_len * max_seq_len + 1, sizeof(char));
  reqs->tok_gens = (int*)calloc(num_reqs * max_seq_len + 1, sizeof(int));
  printf("requests size = %lu B\n",
         ((num_reqs * max_token_len * max_seq_len * sizeof(char) + 1) * 2));
  fflush(stdout);
}

void free_requests(Requests* reqs) {
  free(reqs->str_reqs);
  free(reqs->tok_gens);
}

char* get_str_req_ptr(Requests* reqs, int idx) {
  return reqs->str_reqs + idx * reqs->max_token_len * reqs->max_seq_len;
}

int* get_tok_gen_ptr(Requests* reqs, int idx) {
  return reqs->tok_gens + idx * reqs->max_seq_len;
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
        writeFile << output_tokens[j] << ' ';
      }
      writeFile << "\n";
    }
    writeFile << "\n";
    writeFile.close();
  } else {
    fprintf(stderr, "cannot write the file: %s\n", output_filename);
    exit(EXIT_FAILURE);
  }

  return 0;
}

void warm_up(Transformer* transformer, Tokenizer* tokenizer);
void finish(Transformer* transformer, Tokenizer* tokenizer);
long long inference(Transformer* transformer, Tokenizer* tokenizer, Sampler* sample,
                    Requests* requests);

void getp(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* input_filename,
          char* output_filename, int steps) {
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

  start = time_in_ms();
  warm_up(transformer, tokenizer);
  end = time_in_ms();
  printf("\nwarm up elapsed time(s): %f\n", (double)(end - start) / 1000);
  fflush(stdout);

  start = time_in_ms();
  long long num_gen_tokens = inference(transformer, tokenizer, sampler, &requests);
  end = time_in_ms();
  // Your goal is to achieve best throughput(=reduce elapsed time)!
  fprintf(stdout, "\nelapsed time(s): %f, achieved throughput TPS (tok/s): %f\n",
          (double)(end - start) / 1000, (num_gen_tokens) / (double)(end - start) * 1000);
  fflush(stdout);

  if (EXIT_FAILURE == write_outputfile(output_filename, &requests)) {
    fprintf(stderr, "cannot write output file: %s\n", output_filename);
    exit(EXIT_FAILURE);
  }

  start = time_in_ms();
  finish(transformer, tokenizer);
  end = time_in_ms();
  printf("\nfinish elapsed time(s): %f\n", (double)(end - start) / 1000);
  fflush(stdout);

  // free_requests(&requests);
}

#endif  // GETP_EVAL
