#pragma once
#include "model.hpp"

// TODO: rmv unused
float* forward_hybrid(OssTransformerHybrid* transformer, int token, int pos);
float* forward_hybrid_batch(OssTransformerHybrid* transformer, int* tokens, int pos,
                            int batch_size);

void embed_batch_gpu(float* x_batch, __half* emb_table, int* tokens, int batch_size,
                     int hidden_dim);
void matvec_batch_gpu(float* xout, float* x_batch, __half* w, __half* bias, int batch_size, int n,
                      int d);
void vec_add_vec_batch_gpu(float* a_batch, float* b_batch, float weight, int batch_size, int size);
void split_qkv_batch_gpu(const float* qkv_batch, float* q_batch, float* key_cache,
                         float* value_cache, int batch_size, int head_dim, int n_attn_heads,
                         int n_kv_heads, int layer, int pos, int seq_len);
void rope_batch_gpu(float* x_batch, float* cosv, float* sinv, int batch_size, int n_heads,
                    int head_dim);
void rmsnorm_batch_gpu(float* o_batch, float* x_batch, __half* weight, int batch_size, int size);
void flash_attn_decode_gpu_batch(const float* q_batch, const float* k_cache, const float* v_cache,
                                 const float* mask, const __half* attn_sinks, float* tb_batch,
                                 int batch_size, int pos, int seq_len, int head_dim, int kv_dim,
                                 int kv_mul, int sliding_window, int layer_idx, int n_attn_heads);
void split_gate_up_batch_gpu(float* mlp1_out_batch, float* gate_batch, float* up_batch,
                             int batch_size, int intermediate_dim);
void swiglu_batch_gpu(float* hb_batch, float* hb2_batch, int batch_size, int hidden_dim,
                      float alpha, float swiglu_limit);
