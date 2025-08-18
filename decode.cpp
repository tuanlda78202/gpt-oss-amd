#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "tokenizer.hpp"

void error_usage() {
  fprintf(stderr,
          "Usage:   decode <line_number> [options]\n<line_number> can "
          "be negative to decode all lines\n");
  fprintf(stderr, "Example: decode 0 -i data/output.txt\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr,
          "  -i <string> (optional) input tokens file, default: "
          "data/output.txt\n");
  fprintf(stderr,
          "  -z <string> (optional) optional path to custom tokenizer, "
          "default: tokenizer.bin\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  int line_to_decode;
  std::string input_path = "data/output.txt";
  std::string tokenizer_path = "tokenizer.bin";
  const int vocab_size = 201088;

  if (argc >= 2) {
    line_to_decode = atoi(argv[1]);
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    }  // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    }  // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    }  // must be -x (one dash, one letter)
    // read in the args
    if (strcmp(argv[i], "-i") == 0) {
      input_path = argv[i + 1];
    } else if (strcmp(argv[i], "-z") == 0) {
      tokenizer_path = argv[i + 1];
    } else {
      error_usage();
    }
  }

  Tokenizer tokenizer;
  read_tokenizer(&tokenizer, tokenizer_path.c_str(), vocab_size);

  std::ifstream input_file(input_path);
  std::string line;
  int line_counter = 0;
  while (std::getline(input_file, line)) {
    if (line_to_decode < 0 || line_counter == line_to_decode) {
      std::cout << "Decoding line " << line_counter << ":" << std::endl;
      std::istringstream iss(line);
      int token;
      while (iss >> token) {
        const char* piece = decode_piece(&tokenizer, -1, token);
        safe_printf(piece);  // same as printf("%s", piece), but skips "unsafe" bytes
      }
      std::cout << std::endl;
    }
    line_counter++;
  }

  return 0;
}
