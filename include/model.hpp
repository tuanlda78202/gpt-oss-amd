#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
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

// #define CHECK_RCCL(call)                                                       \
//   do {                                                                         \
//     rcclResult_t status_ = call;                                               \
//     if (status_ != ncclSuccess && status_ != ncclInProgress) {                 \
//       fprintf(stderr, "NCCL error (%s:%d): %s\n", __FILE__, __LINE__,          \
//               ncclGetErrorString(status_));                                    \
//       exit(EXIT_FAILURE);                                                      \
//     }                                                                          \
//   } while (0)

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
    __half* token_embedding_table; // (vocab_size, hidden_dim) (in, out)

    // weights for rmsnorms
    __half* rms_attn_w; // (n_layers, hidden_dim) [attn.norm.scale]
    __half* rms_ffn_w;  // (n_layers, hidden_dim) [mlp.norm.scale]

    // weights for attention [attn.qkv.weight & attn.qkv.bias]
    __half* w_qkv;      // (n_layers, head_dim * n_attn_heads + 2 * head_dim * n_kv_heads,
                        // hidden_dim) where w_q (head_dim * n_attn_heads, hidden_dim)
                        // (out_features, in_features) w_k (head_dim * n_kv_heads,
                        // hidden_dim)  (out_features, in_features) w_v (head_dim *
                        // n_kv_heads, hidden_dim)  (out_features, in_features)
    __half* w_o;        // (n_layers, hidden_dim, head_dim * n_attn_heads)
    __half* b_qkv;      // (n_layers, head_dim * n_attn_heads + 2 * head_dim *
                        // n_kv_heads) (head_dim * n_attn_heads) (head_dim * n_kv_heads)
                        // (head_dim * n_kv_heads)
    __half* b_o;        // (n_layers, hidden_dim)
    __half* attn_sinks; // (n_layers, n_attn_heads)

    // weights for router [mlp.gate.weight & mlp.gate.bias]
    __half* w_router; // (n_layers, hidden_dim, n_experts)
    __half* b_router; // (n_layers, n_experts)

    // weights for MoE [mlp.mlp1_weight & mlp.mlp1_bias & mlp.mlp2_weight &
    // mlp.mlp2_bias] NOTE: gate_up projects from hidden_dim to intermediate_dim,
    // the shape is kinda reverted because the original code use einsum to reduce
    // over hidden_dim

    __half* w_mlp1; // gate_up_proj (n_layers, n_experts, 2 * intermediate_dim,
                    // hidden_dim)
    __half* w_mlp2; // down_proj (n_layers, n_experts, hidden_dim, intermediate_dim)
    __half* b_mlp1; // gate_up proj (n_layers, n_experts, 2 * intermediate_dim)
    __half* b_mlp2; // down_proj (n_layers, n_experts, hidden_dim)

    // final norm [norm.scale]
    __half* rms_out_w; // (hidden_dim, )

    // classifier weights for the logits [unembedding.weight]
    __half* out; // (vocab_size, hidden_dim) (out, in)
} OssTransformerWeightsHalf;

// ! Scratch buffers for forward pass computation
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

    // ! kv cache (largest memory consumer)
    float* key_cache;   // (layer, seq_len, kv_dim)
    float* value_cache; // (layer, seq_len, kv_dim)
    float* mask;
} OssRunState;

typedef struct {
    // current wave of activations
    __half* x;            // activation at current time stamp (hidden_dim, )
    __half* t;            // same, but inside a residual branch (hidden_dim, )
    __half* tb;           // (head_dim * n_attn_heads, )
    __half* tb2;          // (hidden_dim, )
    __half* router_score; // router score (n_experts, )
    __half* topk_v;       // topk expert weights (experts_per_token, )
    int* topk_i;          // topk expert indices (experts_per_token, )
    __half* mlp1_out;
    __half* gate;
    __half* up;
    __half* gate_up;
    __half* e_agg;
    __half* qkv;    // an additional buffer just for convenience (head_dim *
                    // (n_attn_heads + 2 * n_kv_heads), )
    __half* q;      // query (n_attn_heads * head_dim,)
    __half* k;      // key (n_kv_heads * head_dim,)
    __half* v;      // value (n_kv_heads * head_dim,)
    __half* att;    // buffer for scores/attention values (n_heads, seq_len)
    __half* logits; // output logits

    // ! kv cache (largest memory consumer)
    __half* key_cache;   // (layer, seq_len, kv_dim)
    __half* value_cache; // (layer, seq_len, kv_dim)
    __half* mask;
} OssRunStateHalf;

// ! Main Transformer struct
typedef struct {
    OssConfig config;
    OssTransformerWeights weights;
    OssRunState state; // buffers for the "wave" of activations in the forward pass
    int fd;            // file descriptor for memory mapping
    float* data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} OssTransformer;

typedef struct {
    OssConfig config;
    OssTransformerWeightsHalf weights;
    OssRunStateHalf state; // buffers for the "wave" of activations in the forward pass
    int fd;                // file descriptor for memory mapping
    float* data;           // memory mapped data pointer
    ssize_t file_size;     // size of the checkpoint file in bytes
} OssTransformerHalf;

// ! Hybrid Precision Transformer
typedef struct {
    OssConfig config;
    OssTransformerWeightsHalf weights; // FP16 weights for memory efficiency
    OssRunState state;                 // FP32 activations for numerical stability
    int fd;                            // file descriptor for memory mapping
    float* data;                       // memory mapped data pointer
    ssize_t file_size;                 // size of the checkpoint file in bytes
} OssTransformerHybrid;

void copy_large_tensor_streaming(__half** d_ptr, float* h_ptr, size_t total_size,
                                 const char* tensor_name);
void copy_transformer_to_device_hybrid(OssTransformer* t_h, OssTransformerHybrid* t_d);
void free_transformer_on_device_hybrid(OssTransformerHybrid* t_d);
