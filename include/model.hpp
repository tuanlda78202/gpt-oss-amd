#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd)                                                                             \
    do {                                                                                           \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "HIP Error: %s (%d): %s:%d\n", hipGetErrorString(error), error,        \
                    __FILE__, __LINE__);                                                           \
            fflush(stdout);                                                                        \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

typedef struct {
    float value;
    int index;
} OssPair;

// ! Model Hyperparameters
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

    int batch_size; // static batch size for batched inference
} OssConfig;

// ! Learned parameters
typedef struct {
    // token_embedding_table (2D (vocab_size, hidden_dim) -> flat to 1D) - embedding.weight
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
} OssTransformerWeights;

typedef struct {
    // token_embedding_table (2D (vocab_size, hidden_dim) -> flat to 1D) - embedding.weight
    __hip_bfloat16* token_embedding_table; // (vocab_size, hidden_dim) (in, out)

    // weights for rmsnorms
    __hip_bfloat16* rms_attn_w; // (n_layers, hidden_dim) [attn.norm.scale]
    __hip_bfloat16* rms_ffn_w;  // (n_layers, hidden_dim) [mlp.norm.scale]

    // weights for attention [attn.qkv.weight & attn.qkv.bias]
    __hip_bfloat16* w_qkv;      // (n_layers, head_dim * n_attn_heads + 2 * head_dim * n_kv_heads,
                                // hidden_dim) where w_q (head_dim * n_attn_heads, hidden_dim)
                                // (out_features, in_features) w_k (head_dim * n_kv_heads,
                                // hidden_dim)  (out_features, in_features) w_v (head_dim *
                                // n_kv_heads, hidden_dim)  (out_features, in_features)
    __hip_bfloat16* w_o;        // (n_layers, hidden_dim, head_dim * n_attn_heads)
    __hip_bfloat16* b_qkv;      // (n_layers, head_dim * n_attn_heads + 2 * head_dim *
                                // n_kv_heads) (head_dim * n_attn_heads) (head_dim * n_kv_heads)
                                // (head_dim * n_kv_heads)
    __hip_bfloat16* b_o;        // (n_layers, hidden_dim)
    __hip_bfloat16* attn_sinks; // (n_layers, n_attn_heads)

    // weights for router [mlp.gate.weight & mlp.gate.bias]
    __hip_bfloat16* w_router; // (n_layers, hidden_dim, n_experts)
    __hip_bfloat16* b_router; // (n_layers, n_experts)

    __hip_bfloat16* w_mlp1; // gate_up_proj (n_layers, n_experts, 2 * intermediate_dim,
                            // hidden_dim)
    __hip_bfloat16* w_mlp2; // down_proj (n_layers, n_experts, hidden_dim, intermediate_dim)
    __hip_bfloat16* b_mlp1; // gate_up proj (n_layers, n_experts, 2 * intermediate_dim)
    __hip_bfloat16* b_mlp2; // down_proj (n_layers, n_experts, hidden_dim)

    // final norm [norm.scale]
    __hip_bfloat16* rms_out_w; // (hidden_dim, )

    // classifier weights for the logits [unembedding.weight]
    __hip_bfloat16* out; // (vocab_size, hidden_dim) (out, in)
} OssTransformerWeightsBFloat16;

// ! Scratch buffers for forward pass computation
typedef struct {
    float* x;            // activation at current time stamp (B, hidden_dim)
    float* t;            // same, but inside a residual branch (B, hidden_dim)
    float* tb;           // (B, head_dim * n_attn_heads)
    float* tb2;          // (B, hidden_dim)
    float* router_score; // router score (B, n_experts)
    float* topk_v;       // topk expert weights (B, experts_per_token)
    int* topk_i;         // topk expert indices (B, experts_per_token)
    float* mlp1_out;     // (B, 2 * intermediate_dim)
    float* gate;         // (B, intermediate_dim)
    float* up;           // (B, intermediate_dim)
    float* gate_up;      // (B, intermediate_dim)
    float* e_agg;        // (B, hidden_dim)
    float* qkv;          // (B, head_dim * (n_attn_heads + 2 * n_kv_heads))
    float* q;            // query (B, n_attn_heads * head_dim)
    float* k;            // key (B, n_kv_heads * head_dim) - used for current step
    float* v;            // value (B, n_kv_heads * head_dim) - used for current step
    float* att;          // buffer for scores/attention values (B, n_heads, seq_len)
    float* logits;       // output logits (B, vocab_size)

    // ! kv cache
    float* key_cache;   // (layer, B, seq_len, kv_dim)
    float* value_cache; // (layer, B, seq_len, kv_dim)
    float* mask;        // (seq_len, seq_len) - shared across batch

    int* d_batch_indices; // (max_batch_size) - persistent GPU batch indices
    int* d_tokens;        // (max_batch_size) - persistent GPU tokens buffer
    int* d_pos_per_token; // (max_batch_size) - persistent GPU positions per token
    float* cos_vals;      // (head_dim/2) - persistent RoPE cos coefficients
    float* sin_vals;      // (head_dim/2) - persistent RoPE sin coefficients

    //  MoE expert-batching scratch buffers
    int* assign_expert;    // [B*K] - expert assignment per token
    int* assign_token;     // [B*K] - token index per assignment
    float* assign_weight;  // [B*K] - router weight per assignment
    int* expert_counts;    // [E] - tokens per expert (temp/reused as counters)
    int* expert_offsets;   // [E+1] - exclusive prefix sum of expert_counts
    int* tokens_flat;      // [B*K] - tokens grouped by expert
    float* weights_flat;   // [B*K] - weights grouped by expert
    float* x_by_expert;    // [B*K, H] - gathered input by expert
    float* mlp1_by_expert; // [B*K, 2*I] - MLP1 output by expert
    float* gate_by_expert; // [B*K, I] - gate values by expert
    float* up_by_expert;   // [B*K, I] - up values by expert
    float* y_by_expert;    // [B*K, H] - final output by expert

    // Attention workspace buffers
    float* fa_partial_O; // Workspace for flash attention partial outputs
    float* fa_partial_m; // Workspace for flash attention partial max values
    float* fa_partial_l; // Workspace for flash attention partial normalizers
} OssRunState;

// ! Main Transformer struct
typedef struct {
    OssConfig config;
    OssTransformerWeights weights;
    OssRunState state; // buffers for the "wave" of activations in the forward pass
    int fd;            // file descriptor for memory mapping
    float* data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} OssTransformer;

struct PPManager;

// ! Hybrid Precision Transformer
typedef struct {
    OssConfig config;
    OssTransformerWeightsBFloat16 weights; // BF16 weights for memory efficiency
    OssRunState state;                 // FP32 activations for numerical stability
    int fd;                            // file descriptor for memory mapping
    float* data;                       // memory mapped data pointer
    ssize_t file_size;                 // size of the checkpoint file in bytes
    PPManager* pp;
} OssTransformerHybrid;

void copy_large_tensor_streaming(__hip_bfloat16** d_ptr, float* h_ptr, size_t total_size,
                                 const char* tensor_name);
void copy_transformer_to_device(OssTransformer* t_h, OssTransformerHybrid* t_d);
void free_transformer_on_device(OssTransformerHybrid* t_d);
