#include "../include/utils.hpp"
#include "../include/model.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// ! -----------------------------------Memory Management-----------------------------------------
void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = p->head_dim * p->n_kv_heads;
    s->x = reinterpret_cast<float*>(calloc(p->hidden_dim, sizeof(float)));
    s->t = reinterpret_cast<float*>(calloc(p->hidden_dim, sizeof(float)));
    s->tb = reinterpret_cast<float*>(calloc(p->head_dim * p->n_attn_heads, sizeof(float)));
    s->tb2 = reinterpret_cast<float*>(calloc(p->hidden_dim, sizeof(float)));

    s->router_score = reinterpret_cast<float*>(calloc(p->n_experts, sizeof(float)));
    s->topk_v = reinterpret_cast<float*>(calloc(p->experts_per_token, sizeof(float)));
    s->topk_i = reinterpret_cast<int*>(calloc(p->experts_per_token, sizeof(int)));

    s->mlp1_out = reinterpret_cast<float*>(calloc(2 * p->intermediate_dim, sizeof(float)));
    s->gate = reinterpret_cast<float*>(calloc(p->intermediate_dim, sizeof(float)));
    s->up = reinterpret_cast<float*>(calloc(p->intermediate_dim, sizeof(float)));
    s->gate_up = reinterpret_cast<float*>(calloc(p->intermediate_dim, sizeof(float)));
    s->e_agg = reinterpret_cast<float*>(calloc(p->hidden_dim, sizeof(float)));

    s->qkv = reinterpret_cast<float*>(
        calloc(p->head_dim * (p->n_attn_heads + 2 * p->n_kv_heads), sizeof(float)));
    s->q = reinterpret_cast<float*>(calloc(p->n_attn_heads * p->head_dim, sizeof(float)));

    s->key_cache =
        reinterpret_cast<float*>(calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)));
    s->value_cache =
        reinterpret_cast<float*>(calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)));
    s->att = reinterpret_cast<float*>(calloc(p->n_attn_heads * p->seq_len, sizeof(float)));
    s->logits = reinterpret_cast<float*>(calloc(p->vocab_size, sizeof(float)));
    s->mask = p->sliding_window > 0
                  ? reinterpret_cast<float*>(calloc(p->seq_len * p->seq_len, sizeof(float)))
                  : NULL;

    // ensure all mallocs went fine
    if (!s->x || !s->t || !s->tb || !s->tb2 || !s->qkv || !s->q || !s->key_cache ||
        !s->value_cache || !s->att || !s->logits || (p->sliding_window > 0 && !s->mask) ||
        !s->e_agg) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    // initialize mask
    for (int i = 0; i < p->seq_len; i++) {
        for (int j = 0; j < p->seq_len; j++) {
            if (p->sliding_window > 0 && i - j >= p->sliding_window) {
                s->mask[i * p->seq_len + j] = -INFINITY; // Sliding window mask
            }
        }
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->t);
    free(s->tb);
    free(s->tb2);

    free(s->router_score);
    free(s->topk_v);
    free(s->topk_i);

    free(s->mlp1_out);
    free(s->gate);
    free(s->up);
    free(s->gate_up);
    free(s->e_agg);

    free(s->qkv);
    free(s->q);

    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    if (s->mask)
        free(s->mask);
}

void memory_map_weights(TransformerWeights* w, Config* cfg, float* ptr) {
    int head_dim = cfg->head_dim;
    int n_layers = cfg->n_layers;
    int n_experts = cfg->n_experts;
    w->token_embedding_table = ptr;
    ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
    w->out = ptr; // unembedding
    ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
    w->rms_attn_w = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim;
    w->rms_ffn_w = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim;
    w->rms_out_w = ptr;
    ptr += 1ll * cfg->hidden_dim;
    // hey it's qkvqkv, not qqkkvv
    w->w_qkv = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim *
           (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
    w->b_qkv = ptr;
    ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
    w->w_o = ptr;
    ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
    w->b_o = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim;
    w->attn_sinks = ptr;
    ptr += 1ll * n_layers * cfg->n_attn_heads;
    w->w_router = ptr;
    ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
    w->b_router = ptr;
    ptr += 1ll * n_layers * n_experts;
    // hey it's gate_upgate_up, not gategateupup
    w->w_mlp1 = ptr;
    ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;
    w->b_mlp1 = ptr;
    ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;
    w->w_mlp2 = ptr;
    ptr += 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;
    w->b_mlp2 = ptr;
    ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
}

// ! -----------------------------------Model Loading-----------------------------------------
void load_checkpoint(char* ckpt, Config* config, TransformerWeights* weights, int* fd, float** data,
                     ssize_t* file_size) {
    FILE* file = fopen(ckpt, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", ckpt);
        exit(EXIT_FAILURE);
    }

    // read in the config header
    // load sizeof(Config) bytes into config
    if (fread(config, sizeof(Config), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    // figure out the file size
    printf("vocab_size: %d\n", config->vocab_size);
    printf("hidden_dim: %d\n", config->hidden_dim);
    printf("n_experts: %d\n", config->n_experts);
    printf("experts_per_token: %d\n", config->experts_per_token);
    printf("intermediate_dim: %d\n", config->intermediate_dim);
    printf("n_layers: %d\n", config->n_layers);
    printf("head_dim: %d\n", config->head_dim);
    printf("n_attn_heads: %d\n", config->n_attn_heads);
    printf("n_kv_heads: %d\n", config->n_kv_heads);
    printf("max_seq_len: %d\n", config->seq_len);
    printf("init context len: %d\n", config->initial_context_length);
    printf("rope theta: %f\n", config->rope_theta);
    printf("rope_scaling_factor: %f\n", config->rope_scaling_factor);
    printf("sliding window: %d\n", config->sliding_window);
    printf("swiglu_limit: %f\n", config->swiglu_limit);
    fseek(file, 0, SEEK_END); // move file pointer to end of file

    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(ckpt, O_RDONLY); // open in read only mode
    if (*fd == -1) {
        fprintf(stderr, "open failed\n");
        exit(EXIT_FAILURE);
    }
    *data = reinterpret_cast<float*>(mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0));
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    float* weights_ptr = *data + sizeof(Config) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer* t, char* ckpt_path) {
    // read in the Config and the Weights from the checkpoint
    load_checkpoint(ckpt_path, &t->config, &t->weights, &t->fd, &t->data,
                    &t->file_size);          // load model
    malloc_run_state(&t->state, &t->config); // allocate buffers
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) {
        munmap(t->data, t->file_size);
    }
    if (t->fd != -1) {
        close(t->fd);
    }

    // free the RunState buffers
    free_run_state(&t->state);
}