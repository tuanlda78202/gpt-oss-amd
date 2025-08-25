#include "../include/forward.hpp"
#include "hip/attention.hip"
#include "hip/matvec.hip"
#include "hip/prim_add.hip"
#include "hip/rmsnorm.hip"
#include "hip/rope.hip"
#include "hip/softmax.hip"
#include "hip/swilglu.hip"
#include "hip/topk.hip"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

// ! FORWARD
float* forward_hip(OssTransformer* transformer, int token, int pos) {
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
        rmsnorm_hip(s->t, x, w->rms_attn_w + 1ll * l * hidden_dim, hidden_dim);

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

        matmul_hip(s->qkv, s->t, w_qkv, hidden_dim,
                   (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

        // add bias
        vecaddvec_hip(s->qkv, b_qkv, 1.0f, (p->n_attn_heads + 2 * p->n_kv_heads) * head_dim);

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
        compute_cos_sin_hip(pos, p->rope_theta, head_dim, p->rope_scaling_factor,
                            p->initial_context_length, ntk_beta, ntk_alpha, cos_vals, sin_vals);
        apply_rotary_emb_hip(s->q, cos_vals, sin_vals, p->n_attn_heads, head_dim);
        apply_rotary_emb_hip(s->k, cos_vals, sin_vals, p->n_kv_heads, head_dim);

        free(cos_vals);
        free(sin_vals);

        // ! multihead attention
        compute_attention_scores_hip(s->q, s->key_cache + loff, s->att, s->mask, pos, p->seq_len,
                                     head_dim, kv_dim, kv_mul, p->sliding_window, l,
                                     p->n_attn_heads);
        add_attention_sink_hip(s->att, w->attn_sinks + l * p->n_attn_heads, pos, p->seq_len,
                               p->n_attn_heads, l);
        softmax_attention_hip(s->att, pos, p->seq_len, p->n_attn_heads);
        weighted_value_accumulation_hip(s->att, s->value_cache + loff, s->tb, pos, p->seq_len,
                                        head_dim, kv_dim, kv_mul, p->n_attn_heads);
        // ! linear: final matmul to get the output of the attention
        float* w_o = w->w_o + 1ll * l * (head_dim * p->n_attn_heads) * hidden_dim;
        float* b_o = w->b_o + 1ll * l * hidden_dim;
        matmul_hip(s->tb2, s->tb, w_o, head_dim * p->n_attn_heads, hidden_dim);

        // add bias b_o
        vecaddvec_hip(s->tb2, b_o, 1.0f, hidden_dim);

        // residual connection back into x
        vecaddvec_hip(x, s->tb2, 1.0f, hidden_dim);

        // ! ffn rmsnorm
        rmsnorm_hip(s->t, x, w->rms_ffn_w + 1ll * l * hidden_dim, hidden_dim);
        printf("ffn rmsnorm\n");
        for (int i = 0; i < 20; i++) {
            printf("%f ", s->t[i]);
        }
        printf("\n");

        // ! --------------------MoE--------------------
        // ! Router
        // Compute router_score
        float* w_router = w->w_router + 1ll * l * hidden_dim * n_experts;
        float* b_router = w->b_router + 1ll * l * n_experts;
        matmul_hip(s->router_score, s->t, w_router, hidden_dim,
                   n_experts); // s->router_score now stores router_score (n_experts, )

        // add bias b_router
        vecaddvec_hip(s->router_score, b_router, 1.0f, n_experts);

        topk_hip(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);

        // Normalize selected experts using softmax or sigmoid
        softmax_hip(s->topk_v, p->experts_per_token); // expert

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
                matmul_hip(s->mlp1_out, s->t, w_mlp1, hidden_dim,
                           2 * p->intermediate_dim); // (2 * intermediate_dim, )
                // add bias b_mlp1
                vecaddvec_hip(s->mlp1_out, b_mlp1, 1.0f, 2 * p->intermediate_dim);

                // Split mlp1_out into gate and up
                for (int j = 0; j < p->intermediate_dim; j++) {
                    s->gate[j] =
                        s->mlp1_out[2 * j]; // even -> gate (controls what information flow through)
                    s->up[j] = s->mlp1_out[2 * j + 1]; // odd -> up projection (feature transform)
                }

                // ! SwiGLU non-linearity (SwiGLU(x) = Swish(gate(x)) ⊙ up(x))
                const float alpha = 1.702f;
                swiglu_hip(s->gate, s->up, s->gate_up, p->intermediate_dim, alpha, p->swiglu_limit);

                // ! final matmul to get the output of the ffn (down project)
                float* w_mlp2 =
                    w->w_mlp2 + 1ll * (l * n_experts + e) * hidden_dim *
                                    p->intermediate_dim; // (out: hidden_dim, in: intermediate_dim)
                float* b_mlp2 = w->b_mlp2 + 1ll * (l * n_experts + e) * hidden_dim;
                matmul_hip(s->tb2, s->gate_up, w_mlp2, hidden_dim,
                           p->intermediate_dim); // (hidden_dim, )

                // add bias b_mlp2
                vecaddvec_hip(s->tb2, b_mlp2, 1.0f, hidden_dim);

                // ! reduce: aggregate topk experts using weighted sum
                vecaddvec_hip(s->e_agg, s->tb2, expert_w, hidden_dim);
            }
        }

        // ! residual connection (before rms2 to after MoE)
        vecaddvec_hip(x, s->e_agg, 1.0f, hidden_dim);
    }

    // ! final rmsnorm
    rmsnorm_hip(x, x, w->rms_out_w, hidden_dim);

    // ! linear: classifier into logits
    matmul_hip(s->logits, x, w->out, hidden_dim, p->vocab_size);

    return s->logits;
}
