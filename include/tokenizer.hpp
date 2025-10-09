#pragma once

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* str;
    int len;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    int* lengths;
    float* scores;
    int vocab_size;
    int max_token_length;
    TokenIndex* sorted_vocab;
    int byte_tokens[256];
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void* a, const void* b);
int find_token_id(Tokenizer* t, const char* s, int len);
int hex_nibble(int c);
int parse_hex_byte_token(const char* s, int len, unsigned char* out);
void read_tokenizer(Tokenizer* t, const char* path, int vocab_size);
void free_tokenizer(Tokenizer* t);
int find_token_bytes(Tokenizer* t, const unsigned char* p, int len);
unsigned get_merge_rank(Tokenizer* t, const unsigned char* piece, int piece_len, const int* parts,
                        int n_parts, int i);
int encode_piece_bytes_bpe(Tokenizer* t, const unsigned char* piece, int piece_len, int* out,
                           int out_cap);
int encode_with_simple_splits(Tokenizer* t, const unsigned char* bytes, int len, int* out,
                              int out_cap);
void encode(Tokenizer* t, const char* text, int bos_id, int eos_id, int* out, int* n_out,
            int max_tokens);
const char* decode_piece(Tokenizer* t, int prev_token, int token);
void safe_printf(const char* s);
