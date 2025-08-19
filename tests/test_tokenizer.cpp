// test_tokenizer.cpp — Minimal encode/roundtrip tester for tokenizer.bin
//
// Build:
//   gcc -O3 -DTESTING -o test_tokenizer test_tokenizer.cpp -lm
//
// Run:
//   ./test_tokenizer -t tokenizer.bin -i "Hello world" [-r]
//   ./test_tokenizer --help
//
// Behavior:
// - Loads tokenizer.bin
// - Encodes the input string with BOS/EOS disabled (-1 each).
// - Prints space-separated token IDs.
// - If -r/--roundtrip is set, decodes pieces back to text and prints it.
//
// Notes:
// - Compatible with export_tokenizer_bin binary layout:
//     int32  max_token_length
//     repeat n_vocab times: float32 score, int32 byte_len, bytes token_bytes
// - We do a quick pass to count vocab size from the file to call
// read_tokenizer().

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.hpp"

// ---------- Exit codes ----------
enum { EXIT_OK = 0, EXIT_USAGE = 2, EXIT_IO = 3, EXIT_BADFILE = 4, EXIT_NOMEM = 5 };

// ---------- CLI defaults ----------
#define DEFAULT_TOKENIZER_PATH "tokenizer.bin"
#define DEFAULT_PROMPT "Hello"
#define DEFAULT_MAX_TOKENS 8192

// ---------- File helpers ----------

static void die_perror(const char* msg, const char* path, int code) {
    if (path)
        fprintf(stderr, "%s '%s': %s\n", msg, path, strerror(errno));
    else
        fprintf(stderr, "%s: %s\n", msg, strerror(errno));
    exit(code);
}

static void die_msg(const char* msg, int code) {
    fprintf(stderr, "%s\n", msg);
    exit(code);
}

// Return both max_token_length and vocab_size by scanning the file once.
// On failure, exits with a descriptive error.
static void get_vocab_info(const char* path, int* out_max_len, int* out_vocab_size) {
    FILE* f = fopen(path, "rb");
    if (!f)
        die_perror("Failed to open", path, EXIT_IO);

    int32_t maxlen_le = 0;
    if (fread(&maxlen_le, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        die_msg("Bad tokenizer header (cannot read max_token_length).", EXIT_BADFILE);
    }

    int vocab = 0;
    for (;;) {
        float score;
        int32_t len_le;
        if (fread(&score, sizeof(float), 1, f) != 1)
            break; // EOF expected after last
        if (fread(&len_le, sizeof(int32_t), 1, f) != 1)
            break; // truncated?
        if (len_le < 0) {
            fclose(f);
            die_msg("Corrupt tokenizer: negative token length.", EXIT_BADFILE);
        }
        if (fseek(f, (long)len_le, SEEK_CUR) != 0) { // skip token bytes
            fclose(f);
            die_msg("Corrupt tokenizer: cannot seek over token bytes.", EXIT_BADFILE);
        }
        vocab++;
    }
    fclose(f);

    if (vocab <= 0)
        die_msg("Tokenizer appears empty or corrupt.", EXIT_BADFILE);

    if (out_max_len)
        *out_max_len = (int)maxlen_le;
    if (out_vocab_size)
        *out_vocab_size = vocab;
}

// ---------- CLI parsing ----------

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s -t <tokenizer.bin> -i <prompt> [-r] [--max-tokens N]\n"
            "Options:\n"
            "  -t, --tokenizer PATH   Path to tokenizer.bin "
            "(default: " DEFAULT_TOKENIZER_PATH ")\n"
            "  -i, --input TEXT       Prompt to encode (default: \"" DEFAULT_PROMPT "\")\n"
            "  -r, --roundtrip        Also decode pieces and print the reconstructed "
            "text\n"
            "      --max-tokens N     Capacity for encode output buffer (default: "
            "%d)\n"
            "  -h, --help             Show this help\n",
            prog, DEFAULT_MAX_TOKENS);
}

typedef struct {
    const char* tokenizer_path;
    const char* prompt;
    int roundtrip;
    int max_tokens;
} Options;

static void parse_args(int argc, char** argv, Options* opt) {
    opt->tokenizer_path = DEFAULT_TOKENIZER_PATH;
    opt->prompt = DEFAULT_PROMPT;
    opt->roundtrip = 0;
    opt->max_tokens = DEFAULT_MAX_TOKENS;

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (strcmp(arg, "-t") == 0 || strcmp(arg, "--tokenizer") == 0) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                exit(EXIT_USAGE);
            }
            opt->tokenizer_path = argv[++i];
        } else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--input") == 0) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                exit(EXIT_USAGE);
            }
            opt->prompt = argv[++i];
        } else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--roundtrip") == 0) {
            opt->roundtrip = 1;
        } else if (strcmp(arg, "--max-tokens") == 0) {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                exit(EXIT_USAGE);
            }
            opt->max_tokens = atoi(argv[++i]);
            if (opt->max_tokens <= 0)
                die_msg("Invalid --max-tokens value.", EXIT_USAGE);
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            exit(EXIT_OK);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            print_usage(argv[0]);
            exit(EXIT_USAGE);
        }
    }
}

// ---------- Main ----------

int main(int argc, char** argv) {
    Options opt;
    parse_args(argc, argv, &opt);

    int max_token_len = 0;
    int vocab_size = 0;
    get_vocab_info(opt.tokenizer_path, &max_token_len, &vocab_size);

    Tokenizer tok;
    // read_tokenizer() should validate file structure internally.
    read_tokenizer(&tok, opt.tokenizer_path, vocab_size);

    int* ids = (int*)malloc((size_t)opt.max_tokens * sizeof(int));
    if (!ids)
        die_msg("Out of memory allocating token buffer.", EXIT_NOMEM);

    int n_ids = 0;
    // Disable BOS/EOS: we only test pure BPE behavior by default.
    encode(&tok, opt.prompt, /*bos_id=*/-1, /*eos_id=*/-1, ids, &n_ids, opt.max_tokens);

    // Print token IDs
    for (int i = 0; i < n_ids; i++) {
        if (i)
            printf(" ");
        printf("%d", ids[i]);
    }
    printf("\n");

    if (opt.roundtrip) {
        // Decode piece-by-piece using (prev -> curr) transitions for faithful
        // joins.
        for (int i = 0; i < n_ids; i++) {
            int prev = (i == 0) ? -1 : ids[i - 1];
            const char* piece =
                decode_piece(&tok, prev, ids[i]); // owned by tokenizer; printing is safe
            safe_printf(piece);
        }
        printf("\n");
    }

    free(ids);
    free_tokenizer(&tok);
    return EXIT_OK;
}
