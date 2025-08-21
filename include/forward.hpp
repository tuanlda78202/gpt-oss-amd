#pragma once
#include "model.hpp"

// ! -----------------------------------LLMs-----------------------------------
// forward pass on CPU

void rmsnorm(float* o, float* x, float* weight, int size);

void softmax(float* x, int size);

void matmul(float* xout, float* x, float* w, int n, int d);

// Pair struct to store score and original index
typedef struct {
    float value;
    int index;
} Pair;

int compare_desc(const void* a, const void* b);

void topk(float* topk_values, int* topk_indices, float* router_score, int num_experts,
          int experts_per_token);

// ! RoPE
void compute_concentration_and_inv_freq(float base, int head_dim, float scaling_factor,
                                        float initial_context_length, float ntk_beta,
                                        float ntk_alpha, float* concentration_out,
                                        float* inv_freq_out // length head_dim/2
);

void compute_cos_sin(int pos, // position index
                     float base, int head_dim, float scaling_factor, float initial_context_length,
                     float ntk_beta, float ntk_alpha,
                     float* cos_out, // shape: head_dim/2
                     float* sin_out  // shape: head_dim/2
);

void apply_rotary_emb(float* x, float* cos, float* sin, int n_heads, int head_dim);

// ! FORWARD
float* forward(Transformer* transformer, int token, int pos);