#pragma once
#include "BLAS.hip"
#include <cmath>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdint.h>
#include <type_traits>
#include <vector>

#if defined(__HIP_DEVICE_COMPILE__)
template <typename T>
__device__ __forceinline__ T nt_load(const T* ptr) {
    using base_t = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<base_t, float> || std::is_same_v<base_t, double> ||
                  std::is_same_v<base_t, int32_t> || std::is_same_v<base_t, uint32_t> ||
                  std::is_same_v<base_t, int64_t> || std::is_same_v<base_t, uint64_t>) {
        return __builtin_nontemporal_load(ptr);
    } else {
        return *ptr;
    }
}

template <typename T>
__device__ __forceinline__ void nt_store(T* ptr, T value) {
    using base_t = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<base_t, float> || std::is_same_v<base_t, double> ||
                  std::is_same_v<base_t, int32_t> || std::is_same_v<base_t, uint32_t> ||
                  std::is_same_v<base_t, int64_t> || std::is_same_v<base_t, uint64_t>) {
        __builtin_nontemporal_store(value, ptr);
    } else {
        *ptr = value;
    }
}
#else
template <typename T>
__device__ __forceinline__ T nt_load(const T* ptr) {
    return *ptr;
}

template <typename T>
__device__ __forceinline__ void nt_store(T* ptr, T value) {
    *ptr = value;
}
#endif

static __device__ __forceinline__ float wave_reduce_sum(float v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off);
    return v;
}

static __device__ __forceinline__ void wave_sync() {
#if defined(__CUDA_ARCH__)
    __syncwarp();
#elif defined(__HIP_DEVICE_COMPILE__)
    __syncthreads();
#endif
}

// ======================= Ring index helpers =======================
static __device__ __forceinline__ bool is_pow2_int(int x) { return (x & (x - 1)) == 0; }
template <bool POW2>
static __device__ __forceinline__ int ring_index(int t, int cap) {
    if constexpr (POW2)
        return t & (cap - 1);
    else
        return (t >= cap) ? (t % cap) : t;
}

// ======================= K/V vectorized access =======================
static inline __device__ float bf16_to_f32_intr(__hip_bfloat16 x) { return __bfloat162float(x); }

static __device__ __forceinline__ __hip_bfloat16 raw_to_bf16(uint16_t bits) {
    union {
        uint16_t u;
        __hip_bfloat16 b;
    } conv;
    conv.u = bits;
    return conv.b;
}

template <typename KV_T, bool USE_MASK>
__global__ void
fa_decode_full_typed(const float* __restrict__ q_batch,                  // (B, H*D)
                     const void* __restrict__ k_cache_base,              // byte addressable
                     const void* __restrict__ v_cache_base,              // byte addressable
                     const float* __restrict__ mask,                     // (T,T) if USE_MASK
                     const __hip_bfloat16* __restrict__ attn_sinks_half, // (H) or nullptr
                     float* __restrict__ tb_batch,                       // (B, H*D)  [OUTPUT]
                     // sizes & layout
                     int B, int seq_len, int head_dim, int kv_dim, int kv_mul, int sliding_window,
                     int layer_idx, int n_attn_heads,
                     // positions, batch mapping, kv layer info
                     const int* __restrict__ d_pos_per_token, // (B)
                     const int* __restrict__ d_batch_indices, // (B)
                     long long B_stride, const long long* __restrict__ d_layer_kv_off,
                     const int* __restrict__ d_layer_kv_cap,
                     const int* __restrict__ d_layer_is_local) {
    // grid: (h over x, b over y). One Wave64 per block.
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x; // 0..63 (we launch 64 threads)

    if (h >= n_attn_heads || b >= B)
        return;

    // Head group index (GQA/MQA)
    const int g = h / kv_mul;

    // Token position for this request
    const int pos_b = d_pos_per_token[b];
    const int Lb = pos_b + 1;

    // ---- pointers for Q and output ----
    const float* __restrict__ q_b = q_batch + (long long)b * (n_attn_heads * head_dim);
    const float* __restrict__ q_head = q_b + (long long)h * head_dim;
    float* __restrict__ o_out =
        tb_batch + (long long)b * (n_attn_heads * head_dim) + (long long)h * head_dim;

    // ---- shared memory for q and O ----
    extern __shared__ float smem[];
    float* __restrict__ q_sh = smem;            // [head_dim]
    float* __restrict__ O_sh = q_sh + head_dim; // [head_dim]

    // Scale Q by 1/sqrt(d)
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int j = lane; j < head_dim; j += warpSize) {
        q_sh[j] = q_head[j] * inv_sqrt_d;
        O_sh[j] = 0.0f;
    }
    wave_sync();

    // ---- shared scalars for online softmax ----
    __shared__ float m_shared, l_shared, scale_old, scale_new, s_curr;
    if (lane == 0) {
        m_shared = -std::numeric_limits<float>::infinity();
        l_shared = 0.0f;
    }
    wave_sync();

    // ---- KV layer addressing ----
    const long long kv_base = d_layer_kv_off[layer_idx];
    const int cap = d_layer_kv_cap[layer_idx];
    const int local = d_layer_is_local[layer_idx];
    const bool cap_pow2 = is_pow2_int(cap);

    const int gb = d_batch_indices[b];
    const long long batch_off_elems = kv_base + (long long)gb * (long long)cap * kv_dim;

    const size_t elem_size = sizeof(KV_T);
    const char* __restrict__ k_layer_base = (const char*)k_cache_base + batch_off_elems * elem_size;
    const char* __restrict__ v_layer_base = (const char*)v_cache_base + batch_off_elems * elem_size;

    // Local sliding window bounds (for even layers if you use local KV)
    const bool local_and_nomask =
        (local != 0) && !USE_MASK && (sliding_window > 0) && ((layer_idx & 1) == 0);
    const int t_window_start = local_and_nomask ? max(0, pos_b - sliding_window + 1) : 0;
    const int t_start = local_and_nomask ? t_window_start : 0;
    const int t_end = Lb;

    // Per-head base offsets and step in BYTES
    const long long head_off_bytes = ((long long)g * head_dim) * elem_size;
    const long long step_bytes = ((long long)kv_dim) * elem_size;

    // One pass (FlashAttention online softmax).
    for (int t = t_start; t < t_end; ++t) {
        int rt;
        if (local != 0) {
            rt = cap_pow2 ? (t & (cap - 1)) : (t % cap);
        } else {
            rt = t;
        }

        const char* __restrict__ kp_bytes =
            k_layer_base + (long long)rt * step_bytes + head_off_bytes;
        const char* __restrict__ vp_bytes =
            v_layer_base + (long long)rt * step_bytes + head_off_bytes;

        const KV_T* __restrict__ k_head = reinterpret_cast<const KV_T*>(kp_bytes);
        const KV_T* __restrict__ v_head = reinterpret_cast<const KV_T*>(vp_bytes);

        // ---- dot(q, k_t) with vectorized streaming loads ----
        float partial = 0.0f;

        if constexpr (std::is_same<KV_T, float>::value) {
            // FP32 path: float4 vectorization
            const float4* __restrict__ k4 = reinterpret_cast<const float4*>(k_head);
            int vecN = head_dim >> 2, rem = head_dim & 3;

            for (int i4 = lane; i4 < vecN; i4 += warpSize) {
                // non-temporal hint (streaming)
                float4 kv = nt_load(&k4[i4]);
                int j = (i4 << 2);
                partial += q_sh[j + 0] * kv.x + q_sh[j + 1] * kv.y + q_sh[j + 2] * kv.z +
                           q_sh[j + 3] * kv.w;
            }
            for (int j = (vecN << 2) + lane; j < (vecN << 2) + rem; j += warpSize) {
                float kj = nt_load(reinterpret_cast<const float*>(k_head) + j);
                partial += q_sh[j] * kj;
            }
        } else {
            // BF16 path: load 2×bf16 as u32 (packed)
            const uint32_t* __restrict__ k2 = reinterpret_cast<const uint32_t*>(k_head);
            int pairs = head_dim >> 1, rem = head_dim & 1;

            for (int p = lane; p < pairs; p += warpSize) {
                uint32_t w = nt_load(k2 + p);
                __hip_bfloat16 lo = raw_to_bf16((uint16_t)(w & 0xFFFF));
                __hip_bfloat16 hi = raw_to_bf16((uint16_t)(w >> 16));
                int j = (p << 1);
                partial += q_sh[j + 0] * bf16_to_f32_intr(lo) + q_sh[j + 1] * bf16_to_f32_intr(hi);
            }
            if (rem) {
                int j = (pairs << 1) + lane;
                if (j < head_dim) {
                    __hip_bfloat16 kb =
                        nt_load(reinterpret_cast<const __hip_bfloat16*>(k_head) + j);
                    partial += q_sh[j] * bf16_to_f32_intr(kb);
                }
            }
        }

        partial = wave_reduce_sum(partial);
        if (lane == 0) {
            float s = partial;
            if constexpr (USE_MASK) {
                s += mask[(long long)pos_b * seq_len + t];
            }
            s_curr = s;
        }
        wave_sync();

        // Skip update if masked to -inf
        if (!isfinite(s_curr))
            continue;

        if (lane == 0) {
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new = fmaxf(m_prev, s_curr);
            const float alpha = __expf(m_prev - m_new);
            const float p = __expf(s_curr - m_new);
            const float l_new = alpha * l_prev + p;

            scale_old = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new = (l_new > 0.0f) ? p / l_new : 0.0f;

            m_shared = m_new;
            l_shared = l_new;
        }
        wave_sync();

        // ---- O update with vectorized V loads (streaming) ----
        if constexpr (std::is_same<KV_T, float>::value) {
            const float4* __restrict__ v4 = reinterpret_cast<const float4*>(v_head);
            int vecN = head_dim >> 2, rem = head_dim & 3;

            for (int i4 = lane; i4 < vecN; i4 += warpSize) {
                float4 vv = nt_load(&v4[i4]);
                int j = (i4 << 2);
                float o0 = scale_old * O_sh[j + 0] + scale_new * vv.x;
                float o1 = scale_old * O_sh[j + 1] + scale_new * vv.y;
                float o2 = scale_old * O_sh[j + 2] + scale_new * vv.z;
                float o3 = scale_old * O_sh[j + 3] + scale_new * vv.w;
                O_sh[j + 0] = o0;
                O_sh[j + 1] = o1;
                O_sh[j + 2] = o2;
                O_sh[j + 3] = o3;
            }
            for (int j = (vecN << 2) + lane; j < (vecN << 2) + rem; j += warpSize) {
                float vj = nt_load(reinterpret_cast<const float*>(v_head) + j);
                O_sh[j] = scale_old * O_sh[j] + scale_new * vj;
            }
        } else {
            const uint32_t* __restrict__ v2 = reinterpret_cast<const uint32_t*>(v_head);
            int pairs = head_dim >> 1, rem = head_dim & 1;

            for (int p = lane; p < pairs; p += warpSize) {
                uint32_t w = nt_load(v2 + p);
                __hip_bfloat16 lo = raw_to_bf16((uint16_t)(w & 0xFFFF));
                __hip_bfloat16 hi = raw_to_bf16((uint16_t)(w >> 16));
                int j = (p << 1);
                float v0 = bf16_to_f32_intr(lo);
                float v1 = bf16_to_f32_intr(hi);
                O_sh[j + 0] = scale_old * O_sh[j + 0] + scale_new * v0;
                O_sh[j + 1] = scale_old * O_sh[j + 1] + scale_new * v1;
            }
            if (rem) {
                int j = (pairs << 1) + lane;
                if (j < head_dim) {
                    __hip_bfloat16 vb =
                        nt_load(reinterpret_cast<const __hip_bfloat16*>(v_head) + j);
                    float vj = bf16_to_f32_intr(vb);
                    O_sh[j] = scale_old * O_sh[j] + scale_new * vj;
                }
            }
        }
        wave_sync();
    } // end t

    if (attn_sinks_half != nullptr) {
        float sink_s = __bfloat162float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new = fmaxf(m_prev, sink_s);
            const float a = __expf(m_prev - m_new) * l_prev;
            const float b = __expf(sink_s - m_new);
            const float denom = a + b;
            const float renorm = (denom > 0.0f) ? (a / denom) : 1.0f;

            for (int j = lane; j < head_dim; j += warpSize)
                O_sh[j] *= renorm;
            wave_sync();
        }
    }

    // Vectorize stores
    int vecN = head_dim >> 2, rem = head_dim & 3;
    float4* __restrict__ o4 = reinterpret_cast<float4*>(o_out);
    const float4* __restrict__ s4 = reinterpret_cast<const float4*>(O_sh);

    for (int i4 = lane; i4 < vecN; i4 += warpSize) {
        float4 v = s4[i4];
        nt_store(o4 + i4, v);
    }
    for (int j = (vecN << 2) + lane; j < (vecN << 2) + rem; j += warpSize) {
        nt_store(o_out + j, O_sh[j]);
    }
}

static inline int fa_pow2(int x) {
    int p = 1;
    while (p < x)
        p <<= 1;
    return p;
}

void fa(const float* q_batch,             // (B, H*D)
        const void* k_cache,              // base pointer
        const void* v_cache,              // base pointer
        const float* mask,                // (T, T)
        const __hip_bfloat16* attn_sinks, // (L, H) — pass attn_sinks + layer_idx*H
        float* tb_batch,                  // (B, H*D)
        int B, int seq_len, int head_dim, int kv_dim, int kv_mul, int sliding_window, int layer_idx,
        int n_attn_heads, int kv_cache_is_fp16,
        const int* d_pos_per_token,     // (B)
        const int* d_batch_indices,     // (B)
        long long B_stride,             // elements per (B*T*kv_dim) per layer
        int /*max_pos_in_batch*/,       // unused in new path
        float* /*workspace_partial_O*/, // unused
        float* /*workspace_partial_m*/, // unused
        float* /*workspace_partial_l*/, // unused
        hipStream_t stream, const long long* d_layer_kv_off, const int* d_layer_kv_cap,
        const int* d_layer_is_local) {
    if (B <= 0 || n_attn_heads <= 0 || head_dim <= 0)
        return;

    const int tpb = 64; // Wave64 on MI250
    const size_t shmem = 2ull * (size_t)head_dim * sizeof(float);

    const __hip_bfloat16* attn_sinks_head =
        (attn_sinks != nullptr) ? (attn_sinks + (long long)layer_idx * n_attn_heads) : nullptr;

    // Decide compile-time mask specialization for this layer (even layers with SWA)
    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0) && (mask != nullptr);

    dim3 grid(n_attn_heads, B), block(tpb);

    if (kv_cache_is_fp16) {
        if (use_mask) {
            hipLaunchKernelGGL((fa_decode_full_typed<__hip_bfloat16, true>), grid, block, shmem,
                               stream, q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                               B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                               n_attn_heads, d_pos_per_token, d_batch_indices, (long long)B_stride,
                               d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        } else {
            hipLaunchKernelGGL(
                (fa_decode_full_typed<__hip_bfloat16, false>), grid, block, shmem, stream, q_batch,
                k_cache, v_cache, nullptr, attn_sinks_head, tb_batch, B, seq_len, head_dim, kv_dim,
                kv_mul, sliding_window, layer_idx, n_attn_heads, d_pos_per_token, d_batch_indices,
                (long long)B_stride, d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        }
    } else {
        if (use_mask) {
            hipLaunchKernelGGL((fa_decode_full_typed<float, true>), grid, block, shmem, stream,
                               q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch, B,
                               seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                               n_attn_heads, d_pos_per_token, d_batch_indices, (long long)B_stride,
                               d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        } else {
            hipLaunchKernelGGL((fa_decode_full_typed<float, false>), grid, block, shmem, stream,
                               q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch, B,
                               seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx,
                               n_attn_heads, d_pos_per_token, d_batch_indices, (long long)B_stride,
                               d_layer_kv_off, d_layer_kv_cap, d_layer_is_local);
        }
    }

    CHECK_HIP(hipGetLastError());
}
