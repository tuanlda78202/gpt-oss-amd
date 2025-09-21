/* Inference for gpt-oss model in pure C */

#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "tokenizer.hpp"

// ----------------------------------------------------------------------------

typedef struct {
    // Model Config
    int vocab_size; // vocabulary size
    int hidden_dim; // model dim
    // MLP Config
    int n_experts;         // number of experts
    int experts_per_token; // num top-k
    int intermediate_dim;  // for ffn layers
    int n_layers;          // num hidden layers
    // Attention Config
    int head_dim;               // head dimension
    int n_attn_heads;           // number of query heads
    int n_kv_heads;             // number of key/value heads (can be < query heads because of
                                // MQA)
    int seq_len;                // max sequence length e.g., 1024
    int initial_context_length; // e.g., 4096
    float rope_theta;           // rope theta e.g., 150000.0
    float rope_scaling_factor;  // e.g., 32.0
    int sliding_window;         // e.g., 128
    float swiglu_limit;         // e.g., 7.0
} Config;

typedef struct {
    // token_embedding_table - embedding.weight
    float* token_embedding_table; // (vocab_size, hidden_dim) (in, out)
    // weights for rmsnorms
    float* rms_attn_w; // (n_layers, hidden_dim) [attn.norm.scale]
    float* rms_ffn_w;  // (n_layers, hidden_dim) [mlp.norm.scale]
    // weights for attention [attn.qkv.weight & attn.qkv.bias]
    float* w_qkv;      // (n_layers, head_dim * n_attn_heads + 2 * head_dim * n_kv_heads,
                       // hidden_dim) where w_q (head_dim * n_attn_heads, hidden_dim)
                       // (out_features, in_features) w_k (head_dim * n_kv_heads,
                       // hidden_dim)  (out_features, in_features) w_v (head_dim *
                       // n_kv_heads, hidden_dim)  (out_features, in_features)
    float* w_o;        // (n_layers, hidden_dim, head_dim * n_attn_heads)
    float* b_qkv;      // (n_layers, head_dim * n_attn_heads + 2 * head_dim *
                       // n_kv_heads) (head_dim * n_attn_heads) (head_dim * n_kv_heads)
                       // (head_dim * n_kv_heads)
    float* b_o;        // (n_layers, hidden_dim)
    float* attn_sinks; // (n_layers, n_attn_heads)
    // weights for router [mlp.gate.weight & mlp.gate.bias]
    float* w_router; // (n_layers, hidden_dim, n_experts)
    float* b_router; // (n_layers, n_experts)
    // weights for MoE [mlp.mlp1_weight & mlp.mlp1_bias & mlp.mlp2_weight &
    // mlp.mlp2_bias] NOTE: gate_up projects from hidden_dim to intermediate_dim,
    // the shape is kinda reverted because the original code use einsum to reduce
    // over hidden_dim
    float* w_mlp1; // gate_up_proj (n_layers, n_experts, 2 * intermediate_dim,
                   // hidden_dim)
    float* w_mlp2; // down_proj (n_layers, n_experts, hidden_dim, intermediate_dim)
    float* b_mlp1; // gate_up proj (n_layers, n_experts, 2 * intermediate_dim)
    float* b_mlp2; // down_proj (n_layers, n_experts, hidden_dim)
    // final norm [norm.scale]
    float* rms_out_w; // (hidden_dim, )
    // classifier weights for the logits [unembedding.weight]
    float* out; // (vocab_size, hidden_dim) (out, in)
} TransformerWeights;

typedef struct {
    // current wave of activations
    float* x;            // activation at current time stamp (hidden_dim, )
    float* t;            // same, but inside a residual branch (hidden_dim, )
    float* tb;           // (head_dim * n_attn_heads, )
    float* tb2;          // (hidden_dim, )
    float* router_score; // router score (n_experts, )
    float* topk_v;       // topk expert weights (experts_per_token, )
    int* topk_i;         // topk expert indices (experts_per_token, )
    float* mlp1_out;
    float* gate;
    float* up;
    float* gate_up;
    float* e_agg;
    float* qkv;    // an additional buffer just for convenience (head_dim *
                   // (n_attn_heads + 2 * n_kv_heads), )
    float* q;      // query (n_attn_heads * head_dim,)
    float* k;      // key (n_kv_heads * head_dim,)
    float* v;      // value (n_kv_heads * head_dim,)
    float* att;    // buffer for scores/attention values (n_heads, seq_len)
    float* logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, kv_dim)
    float* value_cache; // (layer, seq_len, kv_dim)
    float* mask;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;    // buffers for the "wave" of activations in the forward pass
    int fd;            // file descriptor for memory mapping
    float* data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

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
    load_checkpoint(ckpt_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
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

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    double ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    double max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    double sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // n := in_features, d := out_features
    // W (out,in) @ x (in,) -> xout (out,)
    // by far the most amount of time is spent inside this little function
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        double val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[1ll * i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// Pair struct to store score and original index
typedef struct {
    float value;
    int index;
} Pair;

// Comparator for descending sort (largest value first)
int compare_desc(const void* a, const void* b) {
    float diff = ((Pair*)b)->value - ((Pair*)a)->value;
    if (diff > 0)
        return 1;
    if (diff < 0)
        return -1;
    return 0;
}

// topk function: returns top-k values and their indices
void topk(float* topk_values, int* topk_indices, float* router_score, int num_experts,
          int experts_per_token) {
    if (num_experts <= 0 || experts_per_token <= 0 || experts_per_token > num_experts) {
        fprintf(stderr, "Invalid parameters: num_experts=%d, experts_per_token=%d\n", num_experts,
                experts_per_token);
        return;
    }
    if (!router_score || !topk_values || !topk_indices) {
        fprintf(stderr, "Null pointer detected in topk\n");
        return;
    }
    // Allocate temp array to store (value, index) pairs
    Pair* pairs = reinterpret_cast<Pair*>(malloc(num_experts * sizeof(Pair)));
    if (!pairs) {
        fprintf(stderr, "Memory allocation failed for pairs\n");
        return;
    }
    for (int i = 0; i < num_experts; ++i) {
        pairs[i].value = router_score[i];
        pairs[i].index = i;
    }

    // Sort in descending order of value
    qsort(pairs, num_experts, sizeof(Pair), compare_desc);

    // Fill output arrays
    for (int i = 0; i < experts_per_token; ++i) {
        topk_values[i] = pairs[i].value;
        topk_indices[i] = pairs[i].index;
    }
    free(pairs);
}

void compute_concentration_and_inv_freq(float base, int head_dim, float scaling_factor,
                                        float initial_context_length, float ntk_beta,
                                        float ntk_alpha, float* concentration_out,
                                        float* inv_freq_out // length head_dim/2
) {
    int d_half = head_dim / 2;

    // freq[i] = base ** (i / head_dim)
    float* freq = (float*)malloc(d_half * sizeof(float));
    for (int i = 0; i < d_half; i++) {
        freq[i] = powf(base, ((float)(2 * i)) / (float)head_dim);
    }

    float concentration;
    if (scaling_factor > 1.0f) {
        // YaRN concentration
        concentration = 0.1f * logf(scaling_factor) + 1.0f;

        // NTK by parts
        float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
        float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);

        assert(0 < low && low < high && high < d_half - 1);

        // interpolation = 1 / (scaling_factor * freq)
        // extrapolation = 1 / freq
        for (int i = 0; i < d_half; i++) {
            float interpolation = 1.0f / (scaling_factor * freq[i]);
            float extrapolation = 1.0f / freq[i];

            float ramp = ((float)i - low) / (high - low);
            if (ramp < 0)
                ramp = 0;
            if (ramp > 1)
                ramp = 1;

            float mask = 1.0f - ramp;
            inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
        }
    } else {
        concentration = 1.0f;
        for (int i = 0; i < d_half; i++) {
            inv_freq_out[i] = 1.0f / freq[i];
        }
    }

    *concentration_out = concentration;

    free(freq);
}

void compute_cos_sin(int pos, // position index
                     float base, int head_dim, float scaling_factor, float initial_context_length,
                     float ntk_beta, float ntk_alpha,
                     float* cos_out, // shape: head_dim/2
                     float* sin_out  // shape: head_dim/2
) {
    int d_half = head_dim / 2;

    // Get concentration + inv_freq
    float concentration;
    float* inv_freq = (float*)malloc(d_half * sizeof(float));

    compute_concentration_and_inv_freq(base, head_dim, scaling_factor, initial_context_length,
                                       ntk_beta, ntk_alpha, &concentration, inv_freq);

    // Compute cos and sin for this position
    for (int j = 0; j < d_half; j++) {
        float val = (float)pos * inv_freq[j];
        cos_out[j] = cosf(val) * concentration;
        sin_out[j] = sinf(val) * concentration;
    }

    free(inv_freq);
}

void apply_rotary_emb(float* x, float* cos, float* sin, int n_heads, int head_dim) {
    int half = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < half; i++) {
            // Indexing: head h, dim i
            float x1 = x[h * head_dim + i];        // first half
            float x2 = x[h * head_dim + half + i]; // second half

            float c = cos[i];
            float s = sin[i];

            float o1 = x1 * c - x2 * s;
            float o2 = x2 * c + x1 * s;

            x[h * head_dim + i] = o1;
            x[h * head_dim + half + i] = o2;
        }
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    float* x = s->x;
    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul =
        p->n_attn_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * hidden_dim;
    memcpy(x, content_row, hidden_dim * sizeof(*x));

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // s->t (hidden_dim, )
        rmsnorm(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // s->qkv = w->w_qkv * s->t = (head_dim * (n_attn_heads + 2 * n_kv_heads),
        // hidden_dim) * (hidden_dim, ) = head_dim * (n_attn_heads + 2 * n_kv_heads)
        float* w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                      (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        float* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        matmul(s->qkv, s->t, w_qkv, hidden_dim, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);
        // add bias
        for (int i = 0; i < (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim; ++i) {
            s->qkv[i] += b_qkv[i];
        }
        // Separate q, k, v
        memcpy(s->q, s->qkv, head_dim * p->n_attn_heads * sizeof(float)); // gate
        memcpy(s->k, s->qkv + head_dim * p->n_attn_heads,
               head_dim * p->n_kv_heads * sizeof(float)); // gate
        memcpy(s->v, s->qkv + head_dim * p->n_attn_heads + head_dim * p->n_kv_heads,
               head_dim * p->n_kv_heads * sizeof(float)); // gate

        // RoPE relative positional encoding: complex-valued rotate q and k in each
        // head Adapted from
        // https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L85
        // RoPE with YaRN scaling adapted from Python code
        float ntk_beta = 32.0f;
        float ntk_alpha = 1.0f;
        float* cos_vals = reinterpret_cast<float*>(malloc((head_dim / 2) * sizeof(float)));
        float* sin_vals = reinterpret_cast<float*>(malloc((head_dim / 2) * sizeof(float)));
        compute_cos_sin(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                        p->initial_context_length, ntk_beta, ntk_alpha, cos_vals, sin_vals);
        apply_rotary_emb(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
        apply_rotary_emb(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);

        free(cos_vals);
        free(sin_vals);

        // multihead attention. iterate over all heads
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_attn_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_dim;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                // GQA
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // calculate the attention score as the dot product of q and k
                double score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_dim);
                // Apply sliding window mask if enabled
                if (p->sliding_window > 0 && (l % 2 == 0)) {
                    score += s->mask[pos * p->seq_len + t];
                }
                // save the score to the attention buffer
                att[t] = score;
            }
            // Add attention sink score
            att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];
            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 2);

            // weighted sum of the values
            float* tb = s->tb + h * head_dim;
            memset(tb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                // GQA
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_dim; i++) {
                    tb[i] += a * v[i];
                }
            }
        }
        // final matmul to get the output of the attention
        float* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        float* b_o = w->b_o + 1ll * l * hidden_dim;
        matmul(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);
        // add bias b_o
        for (int i = 0; i < hidden_dim; i++) {
            s->tb2[i] += b_o[i];
        }

        // residual connection back into x
        for (int i = 0; i < hidden_dim; i++) {
            x[i] += s->tb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

        // MoE
        // Compute router_score
        float* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        float* b_router = w->b_router + 1ll * l * n_experts;
        matmul(s->router_score, s->t, w_router, hidden_dim,
               n_experts); // s->router_score now stores router_score (n_experts, )
        // add bias b_router
        for (int i = 0; i < n_experts; i++) {
            s->router_score[i] += b_router[i];
        }
        // Select top-k experts
        topk(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);
        // Normalize selected experts using softmax or sigmoid
        softmax(s->topk_v, p->experts_per_token); // expert

        // Route the tokens to their corresponding top-k experts
        memset(s->e_agg, 0, hidden_dim * sizeof(float));
        for (int e = 0; e < n_experts; e++) {
            float expert_w = 0;
            int in_topk = 0;
            // Check if expert i is in top-k experts
            for (int idx = 0; idx < p->experts_per_token; idx++) {
                if (s->topk_i[idx] == e) {
                    in_topk = 1;
                    expert_w = s->topk_v[idx];
                    break;
                }
            }

            if (in_topk) {
                float* w_mlp1 =
                    w->w_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim) * hidden_dim;
                float* b_mlp1 = w->b_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);
                matmul(s->mlp1_out, s->t, w_mlp1, hidden_dim,
                       2 * p->intermediate_dim); // (2 * intermediate_dim, )
                for (int i = 0; i < 2 * p->intermediate_dim; i++) {
                    s->mlp1_out[i] += b_mlp1[i];
                }
                // Split mlp1_out into gate and up
                for (int j = 0; j < p->intermediate_dim; j++) {
                    s->gate[j] = s->mlp1_out[2 * j];
                    s->up[j] = s->mlp1_out[2 * j + 1];
                }

                // SwiGLU non-linearity
                const float alpha = 1.702f;
                for (int i = 0; i < p->intermediate_dim; i++) {
                    float val = s->gate[i];
                    float up_val = s->up[i];
                    // Clamping
                    if (val > p->swiglu_limit)
                        val = p->swiglu_limit;
                    if (up_val > p->swiglu_limit)
                        up_val = p->swiglu_limit;
                    if (up_val < -p->swiglu_limit)
                        up_val = -p->swiglu_limit;
                    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                    val *= (1.0f / (1.0f + expf(-alpha * val)));
                    // elementwise multiply with w_gate(x)
                    val *= (up_val + 1.0f); // gpt-oss adds an extra bias of 1 to the up layer
                    s->gate_up[i] = val;
                }

                // final matmul to get the output of the ffn
                float* w_mlp2 =
                    w->w_mlp2 + 1ll * (l * n_experts + e) * hidden_dim *
                                    p->intermediate_dim; // (out: hidden_dim, in: intermediate_dim)
                float* b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + e) * hidden_dim;
                matmul(s->tb2, s->gate_up, w_mlp2, p->intermediate_dim,
                       hidden_dim); // (hidden_dim, )
                for (int i = 0; i < hidden_dim; i++) {
                    s->tb2[i] += b_mlp2[i];
                }

                // aggregate topk experts using weighted sum
                for (int i = 0; i < hidden_dim; i++) {
                    s->e_agg[i] += s->tb2[i] * expert_w;
                }
            }
        }

        // residual connection
        for (int i = 0; i < hidden_dim; i++) {
            x[i] += s->e_agg[i];
        }
    }
    // final rmsnorm
    rmsnorm(x, x, w->rms_out_w, hidden_dim);

    // classifier into logits
    matmul(s->logits, x, w->out, hidden_dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a;
    ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp,
                   unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex =
        reinterpret_cast<ProbIndex*>(malloc(sampler->vocab_size * sizeof(ProbIndex)));
}

void free_sampler(Sampler* sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long* state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next =
                sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, const char* prompt,
              int steps) {
    // <|start|>: 200006
    // <|end|>: 200007
    // <|return|>: 200002
    // <|message|>: 200008
    // <|channel|>: 200005
    // <|constrain|>: 200003
    // <|endoftext|>: 199999

    const char* empty_prompt = "";
    if (prompt == NULL) {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens =
        (int*)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    encode(tokenizer, prompt, -1, -1, prompt_tokens, &num_prompt_tokens,
           transformer->config.initial_context_length);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;               // used to time our code, only initialized after first iteration
    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence

    // print the very first token
    const char* first_piece = decode_piece(tokenizer, 200006, token);
    safe_printf(first_piece);
    fflush(stdout);

    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt
            // token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the EOS (=199999 or =200002) token
        // delimits sequences
        if (next == 199999 || next == 200002) {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        const char* piece = decode_piece(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = time_in_ms();
        }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first
    // iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler,
          const char* cli_user_prompt, const char* cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;             // will store the next token in the sequence
    int token;            // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0; // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt,
                               sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, -1, -1, prompt_tokens, &num_prompt_tokens,
                   transformer->config.initial_context_length);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt
            // token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) {
            user_turn = 1;
        }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            const char* piece = decode_piece(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) {
            printf("\n");
        }
    }
    printf("\n");
    free(prompt_tokens);
}

#include "getp-csrc/getp_eval.cpp"
#include "getp-csrc/getp_run.cpp"

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 1024 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 0.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] "
                    "default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 1024. 0 = "
                    "max_seq_len\n");
    fprintf(stderr, "  -i <string> input file in getp mode or input prompt in other modes\n");
    fprintf(stderr, "  -o <string> output file in getp mode\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|getp, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    // default parameters
    char* checkpoint_path = NULL; // e.g. out/model.bin
    const char* tokenizer_path = "tokenizer.bin";
    float temperature = 0.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;        // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 1024;         // number of steps to run for
    char* prompt = NULL;      // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    const char* mode = "generate";   // generate|chat
    char* system_prompt = NULL;      // the (optional) system prompt to use in chat mode
    char* input_filename = NULL;
    char* output_filename = NULL;

    // poor man's C argparse so we can override the defaults above from the
    // command line
    if (argc >= 2) {
        checkpoint_path = argv[1];
    } else {
        error_usage();
    }
    for (int i = 2; i < argc; i += 2) {
        // do some basic validation
        if (i + 1 >= argc) {
            error_usage();
        } // must have arg after flag
        if (argv[i][0] != '-') {
            error_usage();
        } // must start with dash
        if (strlen(argv[i]) != 2) {
            error_usage();
        } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') {
            temperature = atof(argv[i + 1]);
        } else if (argv[i][1] == 'p') {
            topp = atof(argv[i + 1]);
        } else if (argv[i][1] == 's') {
            rng_seed = atoi(argv[i + 1]);
        } else if (argv[i][1] == 'n') {
            steps = atoi(argv[i + 1]);
        } else if (argv[i][1] == 'i') {
            prompt = argv[i + 1];
            input_filename = argv[i + 1];
        } else if (argv[i][1] == 'z') {
            tokenizer_path = argv[i + 1];
        } else if (argv[i][1] == 'm') {
            mode = argv[i + 1];
        } else if (argv[i][1] == 'y') {
            system_prompt = argv[i + 1];
        } else if (argv[i][1] == 'o') {
            output_filename = argv[i + 1];
        } else {
            error_usage();
        }
    }

    // parameter validation/overrides
    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0)
        temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp)
        topp = 0.9;
    if (steps < 0)
        steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    read_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else if (strcmp(mode, "getp") == 0) {
        getp(&transformer, &tokenizer, &sampler, input_filename, output_filename, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}

#endif /* TESTING */
