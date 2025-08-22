#include "../include/forward.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

// ! -----------------------------------LLMs-----------------------------------
void rmsnorm_cpu(float* o, float* x, float* weight, int size) {
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

void softmax_cpu(float* x, int size) {
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

void matmul_cpu(float* xout, float* x, float* w, int n, int d) {
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

// Comparator for descending sort (largest value first)
int compare_desc_cpu(const void* a, const void* b) {
    float diff = ((OssPair*)b)->value - ((OssPair*)a)->value;
    if (diff > 0)
        return 1;
    if (diff < 0)
        return -1;
    return 0;
}

// ! topk: returns top-k values and their indices (expert selection)
void topk_cpu(float* topk_values, int* topk_indices, float* router_score, int num_experts,
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
    OssPair* pairs = reinterpret_cast<OssPair*>(malloc(num_experts * sizeof(OssPair)));
    if (!pairs) {
        fprintf(stderr, "Memory allocation failed for pairs\n");
        return;
    }
    for (int i = 0; i < num_experts; ++i) {
        pairs[i].value = router_score[i];
        pairs[i].index = i;
    }

    // Sort in descending order of value
    qsort(pairs, num_experts, sizeof(OssPair), compare_desc_cpu);

    // Fill output arrays
    for (int i = 0; i < experts_per_token; ++i) {
        topk_values[i] = pairs[i].value;
        topk_indices[i] = pairs[i].index;
    }
    free(pairs);
}

// ! RoPE
void compute_concentration_and_inv_freq_cpu(float base, int head_dim, float scaling_factor,
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

void compute_cos_sin_cpu(int pos, // position index
                         float base, int head_dim, float scaling_factor,
                         float initial_context_length, float ntk_beta, float ntk_alpha,
                         float* cos_out, // shape: head_dim/2
                         float* sin_out  // shape: head_dim/2
) {
    int d_half = head_dim / 2;

    // Get concentration + inv_freq
    float concentration;
    float* inv_freq = (float*)malloc(d_half * sizeof(float));

    compute_concentration_and_inv_freq_cpu(base, head_dim, scaling_factor, initial_context_length,
                                           ntk_beta, ntk_alpha, &concentration, inv_freq);

    // Compute cos and sin for this position
    for (int j = 0; j < d_half; j++) {
        float val = (float)pos * inv_freq[j];
        cos_out[j] = cosf(val) * concentration;
        sin_out[j] = sinf(val) * concentration;
    }

    free(inv_freq);
}

void apply_rotary_emb_cpu(float* x, float* cos, float* sin, int n_heads, int head_dim) {
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

// ! FORWARD
float* forward_cpu(OssTransformer* transformer, int token, int pos) {
    OssConfig* p = &transformer->config;
    OssTransformerWeights* w = &transformer->weights;
    OssRunState* s = &transformer->state;

    float* x = s->x;
    int head_dim = p->head_dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = p->head_dim * p->n_kv_heads;
    int kv_mul =
        p->n_attn_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int intermediate_dim = p->intermediate_dim;
    int n_experts = p->n_experts;

    // ! token embedding: copy the token embedding into x
    float* content_row = w->token_embedding_table + token * hidden_dim;
    memcpy(x, content_row, hidden_dim * sizeof(*x));

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) {
        // ! s->t (hidden_dim, )
        rmsnorm_cpu(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

        // ! kv cache managerment: key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // ! qkv project: s->qkv = w->w_qkv * s->t = (head_dim * (n_attn_heads + 2 * n_kv_heads),
        // hidden_dim) * (hidden_dim, ) = head_dim * (n_attn_heads + 2 * n_kv_heads)
        float* w_qkv = w->w_qkv + 1ll * l * hidden_dim *
                                      (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
        float* b_qkv =
            w->b_qkv + 1ll * l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
      
        matmul_cpu(s->qkv, s->t, w_qkv, hidden_dim,
                   (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

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

        // ! RoPE relative positional encoding: complex-valued rotate q and k in each
        // head Adapted from
        // https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L85
        // RoPE with YaRN scaling adapted from Python code
        float ntk_beta = 32.0f;
        float ntk_alpha = 1.0f;
        float* cos_vals = reinterpret_cast<float*>(malloc((head_dim / 2) * sizeof(float)));
        float* sin_vals = reinterpret_cast<float*>(malloc((head_dim / 2) * sizeof(float)));
        compute_cos_sin_cpu(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                            p->initial_context_length, ntk_beta, ntk_alpha, cos_vals, sin_vals);
        apply_rotary_emb_cpu(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
        apply_rotary_emb_cpu(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);

        free(cos_vals);
        free(sin_vals);

        // ! multihead attention. iterate over all heads
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
                    score += q[i] * k[i]; // ! Dot product
                }
                score /= sqrtf(head_dim);

                // ! Apply sliding window mask if enabled
                if (p->sliding_window > 0 && (l % 2 == 0)) {
                    score += s->mask[pos * p->seq_len + t];
                }

                // save the score to the attention buffer
                att[t] = score;
            }

            // ! Add attention sink score
            att[pos + 1] = w->attn_sinks[l * p->n_attn_heads + h];

            // ! softmax the scores to get attention weights, from 0..pos inclusively
            softmax_cpu(att, pos + 2);

            // ! weighted sum of the values
            float* tb = s->tb + h * head_dim; // concat outputs from all attn_heads
            memset(tb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                // ! GQA
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;

                // get the attention weight for this timestep
                float a = att[t];

                // accumulate the weighted value into xb
                for (int i = 0; i < head_dim; i++) {
                    tb[i] += a * v[i];
                }
            }
        }
        // ! linear: final matmul to get the output of the attention
        float* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        float* b_o = w->b_o + 1ll * l * hidden_dim;
        matmul_cpu(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);

        // add bias b_o
        for (int i = 0; i < hidden_dim; i++) {
            s->tb2[i] += b_o[i];
        }

        // residual connection back into x
        for (int i = 0; i < hidden_dim; i++) {
            x[i] += s->tb2[i];
        }

        // ! ffn rmsnorm
        rmsnorm_cpu(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);

        // ! --------------------MoE--------------------
        // ! Router
        // Compute router_score
        float* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        float* b_router = w->b_router + 1ll * l * n_experts;
        matmul_cpu(s->router_score, s->t, w_router, hidden_dim,
                   n_experts); // s->router_score now stores router_score (n_experts, )

        // add bias b_router
        for (int i = 0; i < n_experts; i++) {
            s->router_score[i] += b_router[i];
        }

        // Select top-k experts
        topk_cpu(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);

        // Normalize selected experts using softmax or sigmoid
        softmax_cpu(s->topk_v, p->experts_per_token); // expert

        // ! expert: Route the tokens to their corresponding top-k experts
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
                // ! Linear 1 (Gated MLP)
                float* w_mlp1 =
                    w->w_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim) * hidden_dim;
                float* b_mlp1 = w->b_mlp1 + 1ll * (l * n_experts + e) * (2 * p->intermediate_dim);
                matmul_cpu(s->mlp1_out, s->t, w_mlp1, hidden_dim,
                           2 * p->intermediate_dim); // (2 * intermediate_dim, )
                for (int i = 0; i < 2 * p->intermediate_dim; i++) {
                    s->mlp1_out[i] += b_mlp1[i];
                }

                // Split mlp1_out into gate and up
                for (int j = 0; j < p->intermediate_dim; j++) {
                    s->gate[j] =
                        s->mlp1_out[2 * j]; // even -> gate (controls what information flow through)
                    s->up[j] = s->mlp1_out[2 * j + 1]; // odd -> up projection (feature transform)
                }

                // ! SwiGLU non-linearity (SwiGLU(x) = Swish(gate(x)) ⊙ up(x))
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

                // ! final matmul to get the output of the ffn (down project)
                float* w_mlp2 =
                    w->w_mlp2 + 1ll * (l * n_experts + e) * hidden_dim *
                                    p->intermediate_dim; // (out: hidden_dim, in: intermediate_dim)
                float* b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + e) * hidden_dim;
                matmul_cpu(s->tb2, s->gate_up, w_mlp2, p->intermediate_dim,
                           hidden_dim); // (hidden_dim, )
                for (int i = 0; i < hidden_dim; i++) {
                    s->tb2[i] += b_mlp2[i];
                }

                // ! reduce: aggregate topk experts using weighted sum
                for (int i = 0; i < hidden_dim; i++) {
                    s->e_agg[i] += s->tb2[i] * expert_w;
                }
            }
        }

        // ! residual connection (before rms2 to after MoE)
        for (int i = 0; i < hidden_dim; i++) {
            x[i] += s->e_agg[i];
        }
    }

    // ! final rmsnorm
    rmsnorm_cpu(x, x, w->rms_out_w, hidden_dim);

    // ! linear: classifier into logits
    matmul_cpu(s->logits, x, w->out, hidden_dim, p->vocab_size);
    return s->logits;
}