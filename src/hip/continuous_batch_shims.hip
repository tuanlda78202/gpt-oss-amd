#pragma once
#include "BLAS.hip"
//#include "split_qkv.hip"
//#include "attention.hip"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>

// =====================================================
// 2.1 Split QKV to KV-cache at each token's own position
// =====================================================

// hip/split_qkv.hip  (add this near your other kernels)
template<int VEC>
__global__ void split_qkv_batch_kernel_mixedpos_vec(
    const float* __restrict__ qkv,     // [B, (Hq+2Hkv)*D]
    float* __restrict__ q_out,         // [B, Hq*D]
    float* __restrict__ key_cache,     // global cache
    float* __restrict__ value_cache,   // global cache
    int B, int head_dim, int n_q_heads, int n_kv_heads,
    int layer, const int* __restrict__ pos_per_token, int seq_len,
    const int* __restrict__ batch_indices, long long B_stride,
    int kv_dim)                         // = head_dim * n_kv_heads
{
    const int t = blockIdx.y;
    const int colv = blockIdx.x * blockDim.x + threadIdx.x;

    if (t >= B) return;

    const int q_row    = n_q_heads * head_dim;
    const int qkv_row  = (n_q_heads + 2*n_kv_heads) * head_dim;

    const long long loff = (long long)layer * (long long)B_stride * (long long)seq_len * (long long)kv_dim;
    const int pos  = pos_per_token[t];
    const int slot = batch_indices[t];

    const float* __restrict__ qkv_t = qkv + (long long)t * qkv_row;
    float*       __restrict__ q_t   = q_out + (long long)t * q_row;

    // Vector copy helpers
    constexpr int V = VEC;
    const int q_col = colv * V;
    const int kv_col = colv * V;

    // 1) Copy Q
    if (q_col < q_row) {
        if constexpr (V == 4) {
            reinterpret_cast<float4*>(q_t)[colv] = reinterpret_cast<const float4*>(qkv_t)[colv];
        } else {
            #pragma unroll
            for (int i = 0; i < V; ++i) {
                int c = q_col + i;
                if (c < q_row) q_t[c] = qkv_t[c];
            }
        }
    }

    // 2) Scatter K and V to the cache at (layer, slot, pos)
    float* __restrict__ kv_dst = key_cache   + loff + (long long)slot * (long long)seq_len * kv_dim + (long long)pos * kv_dim;
    float* __restrict__ vv_dst = value_cache + loff + (long long)slot * (long long)seq_len * kv_dim + (long long)pos * kv_dim;

    const float* __restrict__ k_src = qkv_t + q_row;
    const float* __restrict__ v_src = k_src + kv_dim;

    if (kv_col < kv_dim) {
        if constexpr (V == 4) {
            reinterpret_cast<float4*>(kv_dst)[colv] = reinterpret_cast<const float4*>(k_src)[colv];
            reinterpret_cast<float4*>(vv_dst)[colv] = reinterpret_cast<const float4*>(v_src)[colv];
        } else {
            #pragma unroll
            for (int i = 0; i < V; ++i) {
                int c = kv_col + i;
                if (c < kv_dim) {
                    kv_dst[c] = k_src[c];
                    vv_dst[c] = v_src[c];
                }
            }
        }
    }
}

// host wrapper (REPLACE your current split_qkv_batch_gpu_mixedpos)
inline void split_qkv_batch_gpu_mixedpos(
    float* qkv, float* q_out, float* key_cache, float* value_cache,
    int B, int head_dim, int n_q_heads, int n_kv_heads,
    int layer, const int* d_pos_per_token, int seq_len,
    const int* d_batch_indices, long long B_stride,
    hipStream_t stream, int kv_dim)
{
    const int q_row = n_q_heads * head_dim;
    const int vec = ( (q_row % 4 == 0) && (kv_dim % 4 == 0) ) ? 4 : 1;

    dim3 block(256);
    dim3 grid( ((max(q_row, kv_dim) + (vec-1)) / vec + block.x - 1) / block.x, B );

    if (vec == 4) {
        hipLaunchKernelGGL(split_qkv_batch_kernel_mixedpos_vec<4>, grid, block, 0, stream,
            qkv, q_out, key_cache, value_cache, B, head_dim, n_q_heads, n_kv_heads,
            layer, d_pos_per_token, seq_len, d_batch_indices, B_stride, kv_dim);
    } else {
        hipLaunchKernelGGL(split_qkv_batch_kernel_mixedpos_vec<1>, grid, block, 0, stream,
            qkv, q_out, key_cache, value_cache, B, head_dim, n_q_heads, n_kv_heads,
            layer, d_pos_per_token, seq_len, d_batch_indices, B_stride, kv_dim);
    }
    CHECK_HIP(hipGetLastError());
}


// =====================================================
// 2.2 RoPE per token (compute cos/sin in-kernel for each token)
// =====================================================

// device: compute (cos,sin) for one dim 'i' given pos and rope params
__device__ __forceinline__
void rope_cs_for_dim(int i, int head_dim,
                     float base, float scaling_factor,
                     float initial_ctx_len, float ntk_beta, float ntk_alpha,
                     float pos, float& c, float& s)
{
    // Reuse your compute_cosin_kernel logic (condensed):
    const float d_half = (float)(head_dim >> 1);
    const float two_over_hd = 2.0f / (float)head_dim;
    const float inv_freq_base = -two_over_hd * (float)i; // exponent for base^(...)

    float inv_freq = powf(base, inv_freq_base);
    float concentration = 1.0f;

    if (scaling_factor > 1.0f) {
        concentration = 0.1f * logf(scaling_factor) + 1.0f;

        const float logb = logf(base);
        const float two_pi = 6.28318530717958647692f;
        const float low  = d_half * logf(initial_ctx_len / (ntk_beta  * two_pi)) / logb;
        const float high = d_half * logf(initial_ctx_len / (ntk_alpha * two_pi)) / logb;

        // ramp in [0,1] across i
        float ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - low) / (high - low)));

        // interpolation vs extrapolation mix
        float interpolation = 1.0f / (scaling_factor * powf(base, (2.0f * (float)i) / (float)head_dim));
        float extrapolation = 1.0f / powf(base, (2.0f * (float)i) / (float)head_dim);
        inv_freq = ramp * interpolation + (1.0f - ramp) * extrapolation;
    }

    float theta = pos * inv_freq;
    float ss, cc;  // hip has sincosf
    sincosf(theta, &ss, &cc);
    c = cc * concentration;
    s = ss * concentration;
}

// hip/continuous_batch_shims.hip  (REPLACE the kernel body’s inner loops)
__global__ void rope_qk_fused_batch_kernel_mixed(
    float* __restrict__ q_batch,
    float* __restrict__ key_cache,
    int B, int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, int kv_dim, int layer,
    const int* __restrict__ pos_per_token,
    const int* __restrict__ batch_indices,
    long long B_stride,
    float base, float scaling_factor,
    float initial_ctx_len, float ntk_beta, float ntk_alpha)
{
    const int half = head_dim >> 1;
    const int t    = blockIdx.y;  // token
    const int i0   = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B || i0 >= half) return;

    const int pos  = pos_per_token[t];
    const int slot = batch_indices[t];

    // compute c,s ONCE for this (i0,pos)
    float c, s;
    rope_cs_for_dim(i0, head_dim, base, scaling_factor, initial_ctx_len, ntk_beta, ntk_alpha, (float)pos, c, s);

    // rotate all Q heads
    float* qrow = q_batch + (long long)t * (long long)(n_q_heads*head_dim);
    #pragma unroll 1
    for (int h = 0; h < n_q_heads; ++h) {
        float* qh = qrow + (long long)h*head_dim;
        float x1 = qh[i0];
        float x2 = qh[half + i0];
        qh[i0]        = fmaf(-x2, s, x1 * c);
        qh[half + i0] = fmaf( x1, s, x2 * c);
    }

    // rotate all K heads at (layer, slot, pos)
    const long long loff  = (long long)layer * (long long)B_stride * (long long)seq_len * (long long)kv_dim;
    float* k_base = key_cache + loff + (long long)slot * (long long)seq_len * kv_dim + (long long)pos * kv_dim;

    #pragma unroll 1
    for (int h = 0; h < n_kv_heads; ++h) {
        float* kh = k_base + (long long)h*head_dim;
        float x1 = kh[i0];
        float x2 = kh[half + i0];
        kh[i0]        = fmaf(-x2, s, x1 * c);
        kh[half + i0] = fmaf( x1, s, x2 * c);
    }
}


// Host wrapper
inline void rope_qk_fused_batch_gpu_mixed(
    float* q_batch, float* key_cache,
    int B, int n_q_heads, int n_kv_heads, int head_dim,
    int seq_len, int kv_dim, int layer,
    const int* d_pos_per_token, const int* d_batch_indices,
    long long B_stride,
    float base, float scaling_factor, float initial_ctx_len, float ntk_beta, float ntk_alpha,
    hipStream_t stream)
{
    const int half = head_dim >> 1;
    const dim3 block(256);
    const dim3 grid((half + block.x - 1) / block.x, B);
    hipLaunchKernelGGL(rope_qk_fused_batch_kernel_mixed, grid, block, 0, stream,
        q_batch, key_cache,
        B, n_q_heads, n_kv_heads, head_dim,
        seq_len, kv_dim, layer,
        d_pos_per_token, d_batch_indices, B_stride,
        base, scaling_factor, initial_ctx_len, ntk_beta, ntk_alpha);
    CHECK_HIP(hipGetLastError());
}

// =====================================================
// 2.3 Attention per token (wrapper uses your single-token attention)
// =====================================================
// === Add after your existing includes/protos in attention.hip ===

// Single-CTA per (head,batch) — mixed positions (uses pos_per_token[b])
__launch_bounds__(256, 2)
__global__ void flash_attn_decode_kernel_batched_mixed(
    const float* __restrict__ q_batch,      // (B, H*D)
    const void*  __restrict__ k_cache_base, // base of K-cache
    const void*  __restrict__ v_cache_base, // base of V-cache
    const float* __restrict__ mask,         // (T, T)
    const __half* __restrict__ attn_sinks_half, // (H)
    float* __restrict__ tb_batch,           // (B, H*D)
    int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    const int* __restrict__ pos_per_token,  // (B)  <-- mixed-pos
    const int* __restrict__ batch_indices,  // (B)  slot per token
    long long B_stride                      // elements per (B*T*kv_dim) per layer
) {
    const int h   = blockIdx.x;
    const int b   = blockIdx.y;
    const int tid = threadIdx.x;
    if (h >= n_attn_heads) return;

    const int g = h / kv_mul;

    const float* q_b  = q_batch   + (long long)b * (n_attn_heads * head_dim);
    float*       tb_b = tb_batch  + (long long)b * (n_attn_heads * head_dim);
    const float* q_head = q_b + (long long)h * head_dim;
    float* tb_head = tb_b + (long long)h * head_dim;

    extern __shared__ float q_sh[];
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int i = tid; i < head_dim; i += blockDim.x) q_sh[i] = q_head[i] * inv_sqrt_d;
    __syncthreads();
    for (int i = tid; i < head_dim; i += blockDim.x) tb_head[i] = 0.0f;

    __shared__ float m_shared, l_shared, scale_old_shared, scale_new_shared, s_shared;
    __shared__ int masked_step;
    if (tid == 0) { m_shared = -3.402823466e+38f; l_shared = 0.0f; }
    __syncthreads();

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0);

    const int gb = batch_indices[b];
    const int pos_b = pos_per_token[b];          // <-- per token position
    const long long layer_stride = 1ll * B_stride * seq_len * kv_dim;
    const long long batch_stride = 1ll * seq_len * kv_dim;

#if USE_FP16_KV
    const __half* k_layer = (const __half*)k_cache_base + (long long)layer_idx * layer_stride + (long long)gb * batch_stride;
    const __half* v_layer = (const __half*)v_cache_base + (long long)layer_idx * layer_stride + (long long)gb * batch_stride;
#else
    const float*  k_layer = (const float* )k_cache_base + (long long)layer_idx * layer_stride + (long long)gb * batch_stride;
    const float*  v_layer = (const float* )v_cache_base + (long long)layer_idx * layer_stride + (long long)gb * batch_stride;
#endif

    for (int t = 0; t <= pos_b; ++t) {
#if USE_FP16_KV
        const __half* __restrict__ k_head_h = k_layer + (long long)t * kv_dim + (long long)g * head_dim;
        float partial = 0.0f;
        int i = tid * 2;
        for (; i + 1 < head_dim; i += blockDim.x * 2) {
            __half2 kh2 = *reinterpret_cast<const __half2*>(&k_head_h[i]);
            float2 kf = __half22float2(kh2);
            partial += q_sh[i] * kf.x + q_sh[i + 1] * kf.y;
        }
        for (; i < head_dim; i += blockDim.x) partial += q_sh[i] * __half2float(k_head_h[i]);
#else
        const float*  __restrict__ k_head_f = k_layer + (long long)t * kv_dim + (long long)g * head_dim;
        float partial = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) partial += q_sh[i] * k_head_f[i];
#endif
        partial = block_reduce_sum(partial);

        if (tid == 0) {
            float s = partial;
            if (use_mask) s += mask[(long long)pos_b * seq_len + t];   // per-token pos
            s_shared = s; masked_step = !isfinite(s);
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
            m_shared = m_new; l_shared = l_new;
        }
        __syncthreads();

#if USE_FP16_KV
        const __half* __restrict__ v_head_h = v_layer + (long long)t * kv_dim + (long long)g * head_dim;
        int j = tid * 2;
        for (; j + 1 < head_dim; j += blockDim.x * 2) {
            __half2 vh2 = *reinterpret_cast<const __half2*>(&v_head_h[j]);
            float2 vf = __half22float2(vh2);
            float o0 = tb_head[j]; float o1 = tb_head[j + 1];
            o0 = scale_old_shared * o0 + scale_new_shared * vf.x;
            o1 = scale_old_shared * o1 + scale_new_shared * vf.y;
            tb_head[j] = o0; tb_head[j + 1] = o1;
        }
        for (; j < head_dim; j += blockDim.x) {
            float o = tb_head[j];
            o = scale_old_shared * o + scale_new_shared * __half2float(v_head_h[j]);
            tb_head[j] = o;
        }
#else
        const float*  __restrict__ v_head_f = v_layer + (long long)t * kv_dim + (long long)g * head_dim;
        for (int j = tid; j < head_dim; j += blockDim.x) {
            float o = tb_head[j];
            o = scale_old_shared * o + scale_new_shared * v_head_f[j];
            tb_head[j] = o;
        }
#endif
    }

    // per-head sink (same math)
    if (attn_sinks_half != nullptr) {
        float sink_s = __half2float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m_shared; const float l_prev = l_shared;
            float m_new = fmaxf(m_prev, sink_s);
            float a = __expf(m_prev - m_new) * l_prev;
            float b2= __expf(sink_s - m_new);
            float denom = a + b2; float renorm = (denom > 0.0f) ? (a / denom) : 1.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) tb_head[i] *= renorm;
        }
    }
}

// Chunked CTA per (head,chunk,batch) — mixed positions
__launch_bounds__(256, 2)
__global__ void flash_attn_decode_chunk_kernel_batched_mixed(
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
    long long B_stride
) {
    const int h = blockIdx.x;
    const int c = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    if (h >= n_attn_heads || c >= n_chunks) return;

    const int g = h / kv_mul;
    const int pos_b = pos_per_token[b];
    const int Lb = pos_b + 1;

    // time chunk for this token
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

    extern __shared__ float q_sh[];
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int i = tid; i < head_dim; i += blockDim.x) q_sh[i] = q_head[i] * inv_sqrt_d;
    __syncthreads();
    for (int i = tid; i < head_dim; i += blockDim.x) O_c[i] = 0.0f;

    __shared__ float m_shared, l_shared, scale_old_shared, scale_new_shared, s_shared;
    __shared__ int masked_step;
    if (tid == 0) { m_shared = -3.402823466e+38f; l_shared = 0.0f; }
    __syncthreads();

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0);

    const int gb = batch_indices[b];
    const long long layer_stride = 1ll * B_stride * seq_len * kv_dim;
    const long long batch_stride = 1ll * seq_len * kv_dim;

#if USE_FP16_KV
    const __half* k_layer = (const __half*)k_cache_base + (long long)layer_idx * layer_stride + (long long)gb * batch_stride;
    const __half* v_layer = (const __half*)v_cache_base + (long long)layer_idx * layer_stride + (long long)gb * batch_stride;
#else
    const float*  k_layer = (const float* )k_cache_base + (long long)layer_idx * layer_stride + (long long)gb * batch_stride;
    const float*  v_layer = (const float* )v_cache_base + (long long)layer_idx * layer_stride + (long long)gb * batch_stride;
#endif

    for (int t = t0; t < t1; ++t) {
#if USE_FP16_KV
        const __half* __restrict__ k_head_h = k_layer + (long long)t * kv_dim + (long long)g * head_dim;
        float partial = 0.0f;
        int i = tid * 2;
        for (; i + 1 < head_dim; i += blockDim.x * 2) {
            __half2 kh2 = *reinterpret_cast<const __half2*>(&k_head_h[i]);
            float2 kf = __half22float2(kh2);
            partial += q_sh[i] * kf.x + q_sh[i + 1] * kf.y;
        }
        for (; i < head_dim; i += blockDim.x) partial += q_sh[i] * __half2float(k_head_h[i]);
#else
        const float*  __restrict__ k_head_f = k_layer + (long long)t * kv_dim + (long long)g * head_dim;
        float partial = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) partial += q_sh[i] * k_head_f[i];
#endif
        partial = block_reduce_sum(partial);

        if (tid == 0) {
            float s = partial;
            if (use_mask) s += mask[(long long)pos_b * seq_len + t];   // per-token pos
            s_shared = s; masked_step = !isfinite(s);
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
            m_shared = m_new; l_shared = l_new;
        }
        __syncthreads();

#if USE_FP16_KV
        const __half* __restrict__ v_head_h = v_layer + (long long)t * kv_dim + (long long)g * head_dim;
        int j = tid * 2;
        for (; j + 1 < head_dim; j += blockDim.x * 2) {
            __half2 vh2 = *reinterpret_cast<const __half2*>(&v_head_h[j]);
            float2 vf = __half22float2(vh2);
            float o0 = O_c[j]; float o1 = O_c[j + 1];
            o0 = scale_old_shared * o0 + scale_new_shared * vf.x;
            o1 = scale_old_shared * o1 + scale_new_shared * vf.y;
            O_c[j] = o0; O_c[j + 1] = o1;
        }
        for (; j < head_dim; j += blockDim.x) {
            float o = O_c[j];
            o = scale_old_shared * o + scale_new_shared * __half2float(v_head_h[j]);
            O_c[j] = o;
        }
#else
        const float*  __restrict__ v_head_f = v_layer + (long long)t * kv_dim + (long long)g * head_dim;
        for (int j = tid; j < head_dim; j += blockDim.x) {
            float o = O_c[j];
            o = scale_old_shared * o + scale_new_shared * v_head_f[j];
            O_c[j] = o;
        }
#endif
    }

    if (tid == 0) {
        partial_m[((long long)b*n_attn_heads + h)*n_chunks + c] = m_shared;
        partial_l[((long long)b*n_attn_heads + h)*n_chunks + c] = l_shared;
    }
}

// Reduce across chunks per (b,h) + apply sink (unchanged)
__launch_bounds__(256, 2)
__global__ void flash_attn_reduce_chunks_kernel_batched(
    const float* __restrict__ partial_O,     // (B,H,C,D)
    const float* __restrict__ partial_m,     // (B,H,C)
    const float* __restrict__ partial_l,     // (B,H,C)
    const __half* __restrict__ attn_sinks_half, // (H)
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
        float sink_s = __half2float(attn_sinks_half[h]);
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

// === Replace your old host wrapper with this mixed-pos batched version ===
static inline int fa_next_pow2(int x){ int p=1; while(p<x) p<<=1; return p; }
static inline int choose_chunk_size_batched(int Lmax, int heads, int target_cta_per_head=8) {
    if (Lmax <= 256) return Lmax;
    int c = 256;
    int chunks = (Lmax + c - 1) / c;
    if (chunks < target_cta_per_head) c = 128;
    return c;
}

void flash_attn_decode_gpu_batch_mixed(
    const float* q_batch,      // (B, H*D)
    const void*  k_cache,      // base pointer
    const void*  v_cache,      // base pointer
    const float* mask,         // (T, T)
    const __half* attn_sinks,  // (L, H) — pass attn_sinks + layer_idx*H
    float* tb_batch,           // (B, H*D)
    int B,
    int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    const int* d_pos_per_token,      // (B)
    const int* d_batch_indices,      // (B)
    long long B_stride,              // elements per (B*T*kv_dim) per layer
    int max_pos_in_batch,            // max(pos[b]) in this batch
    hipStream_t stream)
{
    if (B <= 0 || n_attn_heads <= 0 || head_dim <= 0) return;

    const int Lmax = max_pos_in_batch + 1;
    int chunk_size = choose_chunk_size_batched(Lmax, n_attn_heads);
    int n_chunks   = (Lmax + chunk_size - 1) / chunk_size;

    int tpb = fa_next_pow2(head_dim); if (tpb < 64) tpb = 64; else if (tpb > 256) tpb = 256;
    size_t shmem = (size_t)head_dim * sizeof(float);
    const __half* attn_sinks_head = (attn_sinks != nullptr) ? (attn_sinks + (long long)layer_idx * n_attn_heads) : nullptr;

    if (n_chunks <= 1) {
        dim3 grid(n_attn_heads, B), block(tpb);
        hipLaunchKernelGGL(flash_attn_decode_kernel_batched_mixed, grid, block, shmem, stream,
                           q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                           seq_len, head_dim, kv_dim, kv_mul,
                           sliding_window, layer_idx, n_attn_heads,
                           d_pos_per_token, d_batch_indices, (long long)B_stride);
        CHECK_HIP(hipGetLastError());
        return;
    }

    // Workspace (TODO: reuse allocator if you have one)
    float* partial_O; float* partial_m; float* partial_l;
    CHECK_HIP(hipMalloc(&partial_O, (long long)B * n_attn_heads * n_chunks * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&partial_m, (long long)B * n_attn_heads * n_chunks * sizeof(float)));
    CHECK_HIP(hipMalloc(&partial_l, (long long)B * n_attn_heads * n_chunks * sizeof(float)));

    // Kernel 1: chunks
    {
        dim3 grid1(n_attn_heads, n_chunks, B), block1(tpb);
        hipLaunchKernelGGL(flash_attn_decode_chunk_kernel_batched_mixed, grid1, block1, shmem, stream,
                           q_batch, k_cache, v_cache, mask,
                           partial_O, partial_m, partial_l,
                           seq_len, head_dim, kv_dim, kv_mul,
                           sliding_window, layer_idx, n_attn_heads,
                           chunk_size, n_chunks,
                           d_pos_per_token, d_batch_indices, (long long)B_stride);
    }
    // Kernel 2: reduce
    {
        dim3 grid2(n_attn_heads, B), block2(tpb);
        hipLaunchKernelGGL(flash_attn_reduce_chunks_kernel_batched, grid2, block2, 0, stream,
                           partial_O, partial_m, partial_l,
                           attn_sinks_head, tb_batch,
                           head_dim, n_chunks, n_attn_heads);
    }

    CHECK_HIP(hipFree(partial_O));
    CHECK_HIP(hipFree(partial_m));
    CHECK_HIP(hipFree(partial_l));
    CHECK_HIP(hipGetLastError());
}
