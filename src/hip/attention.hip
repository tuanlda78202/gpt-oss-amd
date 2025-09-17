#pragma once
#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>

__global__ void fa_decode_kernel(
    const float* __restrict__ q_batch,      // (B, H*D)
    const void*  __restrict__ k_cache_base,
    const void*  __restrict__ v_cache_base,
    const float* __restrict__ mask,         // (T, T)
    float* __restrict__ partial_O,          // (B,H,C,D)
    float* __restrict__ partial_m,          // (B,H,C)
    float* __restrict__ partial_l,          // (B,H,C)
    int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    int chunk_size, int n_chunks,
    const int* __restrict__ pos_per_token,  // (B)
    const int* __restrict__ batch_indices,  // (B)
    long long B_stride,
    int kv_cache_is_fp16,
    const long long* __restrict__ d_layer_kv_off,
    const int* __restrict__ d_layer_kv_cap,
    const int* __restrict__ d_layer_is_local
) {
    const int h = blockIdx.x;
    const int c = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    if (h >= n_attn_heads || c >= n_chunks) return;

    const int g = h / kv_mul;
    const int pos_b = pos_per_token[b];
    const int Lb = pos_b + 1;

    const int t0 = c * chunk_size;
    int t1 = t0 + chunk_size; if (t1 > Lb) t1 = Lb;
    if (t0 >= t1) {
        if (tid == 0) {
            partial_m[((long long)b*n_attn_heads + h)*n_chunks + c] = -3.402823466e+38f;
            partial_l[((long long)b*n_attn_heads + h)*n_chunks + c] = 0.0f;
        }
        return;
    }

    // Pointers
    const float* q_b = q_batch + (long long)b * (n_attn_heads * head_dim);
    const float* q_head = q_b + (long long)h * head_dim;
    float* O_c = partial_O + ( ((long long)b*n_attn_heads + h) * n_chunks + c ) * (long long)head_dim;

    // ---- On-chip buffers: q_sh and O_sh ----
    extern __shared__ float smem[];
    float* q_sh = smem;                    // [head_dim]
    float* O_sh = q_sh + head_dim;         // [head_dim]

    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        q_sh[i] = q_head[i] * inv_sqrt_d;
        O_sh[i] = 0.0f;                    // keep O on-chip across the whole chunk
    }
    __syncthreads();

    __shared__ float m_shared, l_shared, scale_old_shared, scale_new_shared, s_shared;
    __shared__ int masked_step;
    if (tid == 0) { m_shared = -3.402823466e+38f; l_shared = 0.0f; }
    __syncthreads();

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0) && (mask != nullptr);

    // Ring indexing
    const long long kv_base = d_layer_kv_off[layer_idx];
    const int       cap     = d_layer_kv_cap[layer_idx];
    const int       local   = d_layer_is_local[layer_idx];

    const int gb = batch_indices[b];
    // Handle both float32 and bfloat16 KV cache types
    const size_t element_size = kv_cache_is_fp16 ? sizeof(__hip_bfloat16) : sizeof(float);
    const long long batch_offset = kv_base + (long long)gb * (long long)cap * kv_dim;
    const void*  k_layer_base = (const void*)((const char*)k_cache_base + batch_offset * element_size);
    const void*  v_layer_base = (const void*)((const char*)v_cache_base + batch_offset * element_size);

    const bool local_no_mask = (local != 0) && (mask == nullptr);
    const int t_window_start = local_no_mask ? max(0, pos_b - sliding_window + 1) : 0;

    const int t_start = local_no_mask ? max(t0, t_window_start) : t0;

    for (int t = t_start; t < t1; ++t) {
        float partial = 0.0f;
        const int ring_t = local ? (t % cap) : t;

        if (kv_cache_is_fp16) {
            const __hip_bfloat16* k_head_bf16 = (const __hip_bfloat16*)k_layer_base + (long long)ring_t * kv_dim + (long long)g * head_dim;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                partial += q_sh[i] * __bfloat162float(k_head_bf16[i]);
            }
        } else {
            const float* k_head_f32 = (const float*)k_layer_base + (long long)ring_t * kv_dim + (long long)g * head_dim;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                partial += q_sh[i] * k_head_f32[i];
            }
        }
        partial = block_reduce_sum(partial);

        if (tid == 0) {
            float s = partial;
            if (use_mask) s += mask[(long long)pos_b * seq_len + t];
            s_shared = s;
            masked_step = !isfinite(s);
        }
        __syncthreads();
        if (masked_step) continue;

        if (tid == 0) {
            const float s = s_shared;
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, s);
            const float alpha  = __expf(m_prev - m_new);
            const float p      = __expf(s - m_new);
            const float l_new  = alpha * l_prev + p;

            scale_old_shared = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new_shared = (l_new > 0.0f) ? p / l_new : 0.0f;

            m_shared = m_new;
            l_shared = l_new;
        }
        __syncthreads();

        for (int j = tid; j < head_dim; j += blockDim.x) {
            float o = O_sh[j];
            float v_val;
            if (kv_cache_is_fp16) {
                const __hip_bfloat16* v_head_bf16 = (const __hip_bfloat16*)v_layer_base + (long long)ring_t * kv_dim + (long long)g * head_dim;
                v_val = __bfloat162float(v_head_bf16[j]);
            } else {
                const float* v_head_f32 = (const float*)v_layer_base + (long long)ring_t * kv_dim + (long long)g * head_dim;
                v_val = v_head_f32[j];
            }
            o = scale_old_shared * o + scale_new_shared * v_val;
            O_sh[j] = o;
        }
        __syncthreads();
    }

    for (int j = tid; j < head_dim; j += blockDim.x)
        O_c[j] = O_sh[j];

    if (tid == 0) {
        partial_m[((long long)b*n_attn_heads + h)*n_chunks + c] = m_shared;
        partial_l[((long long)b*n_attn_heads + h)*n_chunks + c] = l_shared;
    }
}

__global__ void fa_reduce_kernel(
    const float* __restrict__ partial_O,     // (B,H,C,D)
    const float* __restrict__ partial_m,     // (B,H,C)
    const float* __restrict__ partial_l,     // (B,H,C)
    const __hip_bfloat16* __restrict__ attn_sinks_half, // (H)
    float* __restrict__ tb_batch,            // (B,H*D)
    int head_dim, int n_chunks, int n_attn_heads
) {
    const int h   = blockIdx.x;
    const int b   = blockIdx.y;
    const int tid = threadIdx.x;
    if (h >= n_attn_heads) return;

    float m = partial_m[((long long)b*n_attn_heads + h)*n_chunks + 0];
    float l = partial_l[((long long)b*n_attn_heads + h)*n_chunks + 0];

    float* O_acc = tb_batch + (long long)b * (n_attn_heads * head_dim) + (long long)h * head_dim;

    const float* O0 = partial_O + ( ((long long)b*n_attn_heads + h) * n_chunks + 0 ) * (long long)head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) O_acc[i] = O0[i];
    __syncthreads();

    for (int c = 1; c < n_chunks; ++c) {
        const float mc = partial_m[((long long)b*n_attn_heads + h)*n_chunks + c];
        const float lc = partial_l[((long long)b*n_attn_heads + h)*n_chunks + c];
        if (!(isfinite(mc) && lc > 0.0f)) continue;

        const float* Oc = partial_O + ( ((long long)b*n_attn_heads + h) * n_chunks + c ) * (long long)head_dim;

        const float m_new = fmaxf(m, mc);
        const float a = __expf(m  - m_new) * l;
        const float b2= __expf(mc - m_new) * lc;
        const float denom = a + b2;
        const float wA = (denom > 0.0f) ? a  / denom : 0.0f;
        const float wB = (denom > 0.0f) ? b2 / denom : 0.0f;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            float o = wA * O_acc[i] + wB * Oc[i];
            O_acc[i] = o;
        }
        __syncthreads();
        m = m_new; l = denom;
    }

    if (attn_sinks_half != nullptr) {
        float sink_s = __bfloat162float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m; const float l_prev = l;
            const float m_new  = fmaxf(m_prev, sink_s);
            const float a = __expf(m_prev - m_new) * l_prev;
            const float b3= __expf(sink_s - m_new);
            const float denom = a + b3;
            const float renorm = (denom > 0.0f) ? (a / denom) : 1.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) O_acc[i] *= renorm;
        }
    }
}

static inline int fa_pow2(int x){ int p=1; while(p<x) p<<=1; return p; }
static inline int choose_chunk_size(int Lmax, int heads, int target_cta_per_head=8) {
    if (Lmax <= 256) return Lmax;
    int c = 256;
    int chunks = (Lmax + c - 1) / c;
    if (chunks < target_cta_per_head) c = 128;
    return c;
}

void fa(
    const float* q_batch,      // (B, H*D)
    const void*  k_cache,      // base pointer
    const void*  v_cache,      // base pointer
    const float* mask,         // (T, T)
    const __hip_bfloat16* attn_sinks,  // (L, H) — pass attn_sinks + layer_idx*H
    float* tb_batch,           // (B, H*D)
    int B,
    int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    int kv_cache_is_fp16,
    const int* d_pos_per_token,      // (B)
    const int* d_batch_indices,      // (B)
    long long B_stride,              // elements per (B*T*kv_dim) per layer
    int max_pos_in_batch,            // max(pos[b]) in this batch
    float* workspace_partial_O,      // Pre-allocated workspace
    float* workspace_partial_m,      // Pre-allocated workspace
    float* workspace_partial_l,      // Pre-allocated workspace
    hipStream_t stream,
    const long long* d_layer_kv_off,
    const int* d_layer_kv_cap,
    const int* d_layer_is_local)
{
    if (B <= 0 || n_attn_heads <= 0 || head_dim <= 0) return;

    const int Lmax = seq_len;
    int chunk_size = choose_chunk_size(Lmax, n_attn_heads);
    int n_chunks   = (Lmax + chunk_size - 1) / chunk_size;

    int tpb = fa_pow2(head_dim); if (tpb < 64) tpb = 64; else if (tpb > 256) tpb = 256;
    size_t shmem = 2ull * (size_t)head_dim * sizeof(float);
    const __hip_bfloat16* attn_sinks_head = (attn_sinks != nullptr) ? (attn_sinks + (long long)layer_idx * n_attn_heads) : nullptr;

    float* partial_O = workspace_partial_O; // (B,H,C,D)
    float* partial_m = workspace_partial_m; // (B,H,C)
    float* partial_l = workspace_partial_l; // (B,H,C)

    // Kernel 1: per-chunk decode with on-chip O
    {
        dim3 grid1(n_attn_heads, n_chunks, B), block1(tpb);
        hipLaunchKernelGGL(fa_decode_kernel, grid1, block1, shmem, stream,
                           q_batch, k_cache, v_cache, mask,
                           partial_O, partial_m, partial_l,
                           seq_len, head_dim, kv_dim, kv_mul,
                           sliding_window, layer_idx, n_attn_heads,
                           chunk_size, n_chunks,
                           d_pos_per_token, d_batch_indices, (long long)B_stride,
                           kv_cache_is_fp16,
                           d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
    }
    // Kernel 2: reduction across chunks
    {
        dim3 grid2(n_attn_heads, B), block2(tpb);
        hipLaunchKernelGGL(fa_reduce_kernel, grid2, block2, 0, stream,
                           partial_O, partial_m, partial_l,
                           attn_sinks_head, tb_batch,
                           head_dim, n_chunks, n_attn_heads);
    }

    CHECK_HIP(hipGetLastError());
}
