#include "../include/tokenizer.hpp"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int compare_tokens(const void* a, const void* b) {
    const TokenIndex* pa = (const TokenIndex*)a;
    const TokenIndex* pb = (const TokenIndex*)b;
    int minlen = pa->len < pb->len ? pa->len : pb->len;
    int cmp = memcmp(pa->str, pb->str, minlen);
    if (cmp != 0)
        return cmp;
    return pa->len - pb->len;
}

int find_token_id(Tokenizer* t, const char* s, int len) {
    TokenIndex key = {.str = s, .len = len, .id = -1};
    TokenIndex* res = (TokenIndex*)bsearch(&key, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex),
                                           compare_tokens);
    return res ? res->id : -1;
}

int hex_nibble(int c) {
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return -1;
}

int parse_hex_byte_token(const char* s, int len, unsigned char* out) {
    if (len != 6)
        return 0; // expects exactly "<0xHH>"
    if (s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>')
        return 0;

    int h = hex_nibble((unsigned char)s[3]);
    int l = hex_nibble((unsigned char)s[4]);
    if (h < 0 || l < 0)
        return 0;

    *out = (unsigned char)((h << 4) | l);
    return 1;
}

void read_tokenizer(Tokenizer* t, const char* path, int vocab_size) {
    memset(t, 0, sizeof(*t));
    t->vocab_size = vocab_size;

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer open %s failed: %s\n", path, strerror(errno));
        exit(1);
    }

    if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "bad tokenizer header\n");
        exit(1);
    }

    t->vocab = (char**)calloc(vocab_size, sizeof(char*));
    t->lengths = (int*)calloc(vocab_size, sizeof(int));
    t->scores = (float*)calloc(vocab_size, sizeof(float));
    t->sorted_vocab = (TokenIndex*)calloc(vocab_size, sizeof(TokenIndex));
    if (!t->vocab || !t->scores || !t->sorted_vocab || !t->lengths) {
        fprintf(stderr, "OOM tokenizer\n");
        exit(1);
    }

    for (int i = 0; i < vocab_size; i++) {
        float score;
        int len;
        if (fread(&score, sizeof(float), 1, f) != 1) {
            fprintf(stderr, "bad tokenizer file (score)\n");
            exit(1);
        }
        if (fread(&len, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "bad tokenizer file (len)\n");
            exit(1);
        }
        char* buf = (char*)malloc((size_t)len + 1);
        if (!buf) {
            fprintf(stderr, "OOM\n");
            exit(1);
        }
        if ((int)fread(buf, 1, len, f) != len) {
            fprintf(stderr, "bad tokenizer file (str)\n");
            exit(1);
        }
        buf[len] = '\0';

        t->vocab[i] = buf;
        t->lengths[i] = len;
        t->scores[i] = score;
        t->sorted_vocab[i].str = buf;
        t->sorted_vocab[i].len = len;
        t->sorted_vocab[i].id = i;
    }
    fclose(f);

    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);

    char tmp[8];
    for (int b = 0; b < 256; b++) {
        int id = -1;
        snprintf(tmp, sizeof(tmp), "<0x%02X>", b);
        id = find_token_id(t, tmp, (int)strlen(tmp));
        if (id < 0) {
            char one[2] = {(char)b, 0};
            if (b == 9 || b == 10 || b == 13 || (b >= 32 && b <= 126)) {
                id = find_token_id(t, one, 1);
            }
        }
        t->byte_tokens[b] = id;
        t->byte_pieces[b * 2] = (unsigned char)b;
        t->byte_pieces[b * 2 + 1] = '\0';
    }
}

void free_tokenizer(Tokenizer* t) {
    if (!t)
        return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++)
            free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->lengths);
    free(t->scores);
    free(t->sorted_vocab);
}

int find_token_bytes(Tokenizer* t, const unsigned char* p, int len) {
    TokenIndex key = {.str = (const char*)p, .len = len, .id = -1};
    TokenIndex* res = (TokenIndex*)bsearch(&key, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex),
                                           compare_tokens);
    return res ? res->id : -1;
}

unsigned get_merge_rank(Tokenizer* t, const unsigned char* piece, int piece_len, const int* parts,
                        int n_parts, int i) {
    const unsigned RANK_MAX = 0xFFFFFFFFu;
    if (i + 2 >= n_parts)
        return RANK_MAX;

    int start = parts[i];
    int end = parts[i + 2];
    int span = end - start;

    if (span <= 0 || span > t->max_token_length || end > piece_len)
        return RANK_MAX;

    int id = find_token_bytes(t, piece + start, span);
    if (id < 0)
        return RANK_MAX;

    if (t->scores[id] <= -1e29f)
        return RANK_MAX;

    return (unsigned)id;
}

int encode_piece_bytes_bpe(Tokenizer* t, const unsigned char* piece, int piece_len, int* out,
                           int out_cap) {
    if (piece_len <= 0)
        return 0;

    if (piece_len == 1) {
        int id = find_token_bytes(t, piece, 1);
        if (id < 0)
            id = 0;
        if (out_cap > 0)
            out[0] = id;
        return 1;
    }

    int parts_cap = piece_len + 2;
    int* parts = (int*)malloc((size_t)parts_cap * sizeof(int));
    if (!parts) {
        fprintf(stderr, "OOM parts\n");
        exit(1);
    }

    int n_parts = 0;
    for (int i = 0; i < piece_len; i++)
        parts[n_parts++] = i;
    parts[n_parts++] = piece_len;

    while (1) {
        const unsigned RANK_MAX = 0xFFFFFFFFu;
        unsigned best_rank = RANK_MAX;
        int best_i = -1;

        for (int i = 0; i + 2 < n_parts; i++) {
            unsigned r = get_merge_rank(t, piece, piece_len, parts, n_parts, i);
            if (r < best_rank) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i < 0)
            break;

        for (int j = best_i + 1; j + 1 < n_parts; j++)
            parts[j] = parts[j + 1];
        n_parts--;
    }

    int n_out = 0;
    for (int i = 0; i + 1 < n_parts; i++) {
        int start = parts[i];
        int end = parts[i + 1];
        int span = end - start;
        if (span <= 0)
            continue;
        int id = find_token_bytes(t, piece + start, span);
        if (id < 0)
            id = 0;
        if (n_out < out_cap)
            out[n_out] = id;
        n_out++;
    }

    free(parts);
    return n_out;
}

int encode_with_simple_splits(Tokenizer* t, const unsigned char* bytes, int len, int* out,
                              int out_cap) {
    int n_out = 0;
    int i = 0;

    while (i < len) {
        if (bytes[i] >= '0' && bytes[i] <= '9') {
            int j = i + 1;
            while (j < len && bytes[j] >= '0' && bytes[j] <= '9')
                j++;
            int run_len = j - i;
            int k = i;
            while (run_len > 0) {
                int chunk = run_len >= 3 ? 3 : run_len;
                if (n_out < out_cap)
                    n_out +=
                        encode_piece_bytes_bpe(t, bytes + k, chunk, out + n_out, out_cap - n_out);
                else
                    n_out += encode_piece_bytes_bpe(t, bytes + k, chunk, NULL, 0);
                k += chunk;
                run_len -= chunk;
            }
            i = j;
        } else {
            int j = i + 1;
            while (j < len && !(bytes[j] >= '0' && bytes[j] <= '9'))
                j++;
            int span = j - i;
            if (n_out < out_cap)
                n_out += encode_piece_bytes_bpe(t, bytes + i, span, out + n_out, out_cap - n_out);
            else
                n_out += encode_piece_bytes_bpe(t, bytes + i, span, NULL, 0);
            i = j;
        }
    }
    return n_out;
}

void encode(Tokenizer* t, const char* text, int bos_id, int eos_id, int* out, int* n_out,
            int max_tokens) {
    const unsigned char* bytes = (const unsigned char*)text;
    const int len = (int)strlen((const char*)bytes);

    int ntok = 0;
    if (bos_id >= 0 && ntok < max_tokens)
        out[ntok++] = bos_id;

    if (ntok < max_tokens) {
        ntok += encode_with_simple_splits(t, bytes, len, out + ntok, max_tokens - ntok);
        if (ntok > max_tokens)
            ntok = max_tokens;
    }

    if (eos_id >= 0 && ntok < max_tokens)
        out[ntok++] = eos_id;
    *n_out = ntok;
}

const char* decode_piece(Tokenizer* t, int prev_token, int token) {
    (void)prev_token;
    if (token < 0 || token >= t->vocab_size)
        return "";

    const char* piece = t->vocab[token];
    int len = t->lengths[token];

    if (len == 1) {
        static char out[2];
        out[0] = piece[0];
        out[1] = '\0';
        return out;
    }

    unsigned char b;
    if (parse_hex_byte_token(piece, len, &b)) {
        return (char*)&t->byte_pieces[(int)b * 2];
    }

    return (char*)piece;
}

void safe_printf(const char* s) {
    if (!s || !*s)
        return;
    fputs(s, stdout);
}
