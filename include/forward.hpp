#pragma once
#include "model.hpp"

// ! CPU
// ! Layers
void rmsnorm_cpu(float* o, float* x, float* weight, int size);

void softmax_cpu(float* x, int size);

void matmul_cpu(float* xout, float* x, float* w, int n, int d);

// pair struct to store score and original index

int compare_desc_cpu(const void* a, const void* b);

void topk_cpu(float* topk_values, int* topk_indices, float* router_score, int num_experts,
              int experts_per_token);

// ! RoPE
void compute_concentration_and_inv_freq_cpu(float base, int head_dim, float scaling_factor,
                                            float initial_context_length, float ntk_beta,
                                            float ntk_alpha, float* concentration_out,
                                            float* inv_freq_out // length head_dim/2
);

void compute_cos_sin_cpu(int pos, // position index
                         float base, int head_dim, float scaling_factor,
                         float initial_context_length, float ntk_beta, float ntk_alpha,
                         float* cos_out, // shape: head_dim/2
                         float* sin_out  // shape: head_dim/2
);

void apply_rotary_emb_cpu(float* x, float* cos, float* sin, int n_heads, int head_dim);

// ! Forward
float* forward_cpu(OssTransformer* transformer, int token, int pos);
