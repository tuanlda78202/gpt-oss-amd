#pragma once
#include "BLAS.hip"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <type_traits>

template<typename T>
__device__ __forceinline__ T ld_global_cond(const T* p) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                  std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
    }
#endif
    return *p;
}

__device__ __forceinline__ uint32_t ld_global_u32_cond(const uint32_t* p) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_nontemporal_load(p);
#endif
    return *p;
}

template<int WIDTH>
__device__ __forceinline__ float wave_reduce_sum_width(float v) {
#pragma unroll
    for (int off = WIDTH >> 1; off > 0; off >>= 1) {
        v += __shfl_down(v, off, WIDTH);
    }
    return v;
}

static __device__ __forceinline__ void wave_sync() {
#if defined(__CUDA_ARCH__)
    __syncwarp();
#elif defined(__HIP_DEVICE_COMPILE__)
    __syncthreads();
#endif
}

static __device__ __forceinline__ bool is_pow2_int(int x) {
    return (x & (x - 1)) == 0;
}

template<bool POW2>
static __device__ __forceinline__ int ring_index(int t, int cap) {
    if constexpr (POW2)
        return t & (cap - 1);
    else
        return (t >= cap) ? (t % cap) : t;
}

static inline __device__ float bf16_to_f32_intr(__hip_bfloat16 x) {
    return __bfloat162float(x);
}

static __device__ __forceinline__ __hip_bfloat16 raw_to_bf16(uint16_t bits) {
    union {
        uint16_t u;
        __hip_bfloat16 b;
    } conv;
    conv.u = bits;
    return conv.b;
}

__device__ __forceinline__ void unpack_bf16x2(uint32_t packed, float& lo, float& hi) {
    __hip_bfloat16 bf_lo = raw_to_bf16(static_cast<uint16_t>(packed & 0xFFFF));
    __hip_bfloat16 bf_hi = raw_to_bf16(static_cast<uint16_t>(packed >> 16));
    lo = bf16_to_f32_intr(bf_lo);
    hi = bf16_to_f32_intr(bf_hi);
}

__device__ __forceinline__ bool is_aligned_4(const void* p) {
    return ((uintptr_t)p & 3) == 0;
}

// ======================================================================
// FlashAttention-2 decode kernel with MFMA tiling (head_dim == 64)
// - One wave (64 threads) computes a Br x Bc tile with MFMA (Br=Bc=16).
// - Processes all attention heads that share the same KV head (kv_mul).
// - Performs online softmax and O update in-place while streaming V once.
// ======================================================================
template<typename KV_T, bool USE_MASK, int BR_TILE = 16, int BC_TILE = 16>
__global__ void fa2_decode_mfma_head64(
    const float* __restrict__ q_batch,      // (B, H*64)
    const void*  __restrict__ k_cache_base, // base (byte addr)
    const void*  __restrict__ v_cache_base, // base (byte addr)
    const float* __restrict__ mask,         // (T,T) if USE_MASK
    const __hip_bfloat16* __restrict__ attn_sinks_half, // (H) or nullptr
    float* __restrict__ tb_batch,           // (B, H*64)  [OUTPUT]
    int B, int seq_len, int /*head_dim == 64*/, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    const int* __restrict__ d_pos_per_token,  // (B)
    const int* __restrict__ d_batch_indices,  // (B)
    long long /*B_stride*/,                   // unused (layer offsets used)
    const long long* __restrict__ d_layer_kv_off,
    const int*       __restrict__ d_layer_kv_cap,
    const int*       __restrict__ d_layer_is_local,
    int bc_tile_hint)
{
    static_assert(BR_TILE == 16, "This kernel assumes BR_TILE == 16");
    static_assert(BC_TILE == 16, "This kernel assumes BC_TILE == 16");

    constexpr int HEAD_DIM = 64;
    constexpr int MFMA_K = 16;
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

    const int lane_id = threadIdx.x;
    const int lane_x = lane_id % BC_TILE; // column within tile
    const int lane_y = lane_id / BC_TILE; // row-group (4 rows) within tile

    const int kv_head = blockIdx.x; // which KV head this wave handles
    const int b = blockIdx.y;       // batch index
    if (lane_id >= 64 || b >= B) {
        return;
    }

    if (kv_mul > BR_TILE) {
        return; // unsupported configuration for this kernel
    }

    const int rows_active = kv_mul;
    const int h_base = kv_head * kv_mul;
    if (h_base >= n_attn_heads) {
        return;
    }

    const int pos_b = d_pos_per_token[b];
    const int Lb = pos_b + 1;

    const int local = d_layer_is_local[layer_idx];
    const long long kv_base = d_layer_kv_off[layer_idx];
    const int cap = d_layer_kv_cap[layer_idx];
    const bool cap_pow2 = is_pow2_int(cap);

    const bool local_nomask = (local != 0) && !USE_MASK;
    const int t_window_start = local_nomask ? max(0, pos_b - cap + 1) : 0;
    const int t_start = local_nomask ? t_window_start : 0;
    const int t_end = Lb;

    const int gb = d_batch_indices[b];
    const long long batch_off_elems = kv_base + static_cast<long long>(gb) * cap * kv_dim;

    const size_t elem_size = sizeof(KV_T);
    const char* __restrict__ k_layer_base = reinterpret_cast<const char*>(k_cache_base) + batch_off_elems * elem_size;
    const char* __restrict__ v_layer_base = reinterpret_cast<const char*>(v_cache_base) + batch_off_elems * elem_size;

    const long long head_off_bytes = static_cast<long long>(kv_head) * HEAD_DIM * elem_size;
    const long long step_bytes = static_cast<long long>(kv_dim) * elem_size;

    const float* __restrict__ q_block = q_batch + static_cast<long long>(b) * n_attn_heads * HEAD_DIM;
    float* __restrict__ o_block = tb_batch + static_cast<long long>(b) * n_attn_heads * HEAD_DIM;

    const float inv_sqrt_d = rsqrtf(static_cast<float>(HEAD_DIM));

    __shared__ float row_m[BR_TILE];
    __shared__ float row_l[BR_TILE];
    __shared__ float row_prev_m[BR_TILE];
    __shared__ float row_scale[BR_TILE];
    __shared__ float row_inv_l[BR_TILE];
    __shared__ float tile_max[BR_TILE];
    __shared__ float tile_sum[BR_TILE];
    __shared__ float score_tile[BR_TILE * BC_TILE];
    __shared__ float prob_tile[BR_TILE * BC_TILE];
    __shared__ int   tile_rt[BC_TILE];
    __shared__ short v_tile[HEAD_DIM][BC_TILE];

    if (lane_id < BR_TILE) {
        row_m[lane_id] = NEG_INF;
        row_l[lane_id] = 0.0f;
    }
    __syncthreads();

    // Preload Q fragments (bf16) for each row handled by the wave
    i16x4 q_frag[HEAD_DIM / MFMA_K];
#pragma unroll
    for (int kk = 0; kk < HEAD_DIM; kk += MFMA_K) {
        const int frag_idx = kk / MFMA_K;
        i16x4 frag = {0, 0, 0, 0};
        const int row = lane_x;
        if (row < rows_active) {
            const float* __restrict__ q_row = q_block + static_cast<long long>(h_base + row) * HEAD_DIM;
            const int k_base = kk + lane_y * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k_idx = k_base + i;
                float qv = 0.0f;
                if (k_idx < HEAD_DIM) {
                    qv = q_row[k_idx] * inv_sqrt_d;
                }
                frag[i] = bf16_bits(__float2bfloat16(qv));
            }
        }
        q_frag[frag_idx] = frag;
    }

    f4 o_frag0 = {0.f, 0.f, 0.f, 0.f};
    f4 o_frag1 = {0.f, 0.f, 0.f, 0.f};
    f4 o_frag2 = {0.f, 0.f, 0.f, 0.f};
    f4 o_frag3 = {0.f, 0.f, 0.f, 0.f};

    const int bc_try = (bc_tile_hint > 0 && bc_tile_hint <= BC_TILE) ? bc_tile_hint : BC_TILE;

    for (int t0 = t_start; t0 < t_end; t0 += bc_try) {
        int bc = t_end - t0;
        if (bc > bc_try) bc = bc_try;
        if (bc > BC_TILE) bc = BC_TILE;

        const int col = lane_x;
        bool col_active = (col < bc);
        const int t = t0 + col;

        int rt = 0;
        if (col_active) {
            if (local != 0) {
                rt = cap_pow2 ? (t & (cap - 1)) : (t % cap);
            } else {
                rt = t;
            }
            if (lane_y == 0) {
                tile_rt[col] = rt;
            }
        }
        __syncthreads();

        const long long col_off = static_cast<long long>(rt) * step_bytes + head_off_bytes;
        const KV_T* k_col_ptr = col_active ? reinterpret_cast<const KV_T*>(k_layer_base + col_off) : nullptr;

        auto load_k_fragment = [&](const KV_T* ptr, int k_off, bool valid) {
            i16x4 frag = {0, 0, 0, 0};
            if (!valid) {
                return frag;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = k_off + i;
                float val = 0.0f;
                if (idx < HEAD_DIM) {
                    if constexpr (std::is_same<KV_T, float>::value) {
                        val = reinterpret_cast<const float*>(ptr)[idx];
                    } else {
                        val = bf16_to_f32_intr(reinterpret_cast<const __hip_bfloat16*>(ptr)[idx]);
                    }
                }
                frag[i] = bf16_bits(__float2bfloat16(val));
            }
            return frag;
        };

        f4 score_frag = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < HEAD_DIM; kk += MFMA_K) {
            const int frag_idx = kk / MFMA_K;
            const int k_base = kk + lane_y * 4;
            const bool valid = col_active;
            i16x4 b_frag = load_k_fragment(k_col_ptr, k_base, valid);
            score_frag = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(q_frag[frag_idx], b_frag, score_frag, 0, 0, 0);
        }

        float mask_val = 0.0f;
        if constexpr (USE_MASK) {
            if (col_active) {
                mask_val = mask[static_cast<long long>(pos_b) * seq_len + t];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = lane_y * 4 + i;
            float val = score_frag[i];
            if constexpr (USE_MASK) {
                val += mask_val;
            }
            if (!(col_active && row < rows_active)) {
                val = NEG_INF;
            }
            score_frag[i] = val;
            if (row < BR_TILE) {
                score_tile[row * BC_TILE + col] = val;
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float v = score_frag[i];
#pragma unroll
            for (int offset = BC_TILE >> 1; offset > 0; offset >>= 1) {
                v = fmaxf(v, __shfl_down(v, offset, BC_TILE));
            }
            if (lane_x == 0) {
                tile_max[lane_y * 4 + i] = v;
            }
        }
        __syncthreads();

        if (lane_x == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = lane_y * 4 + i;
                const float prev_m = row_m[row];
                row_prev_m[row] = prev_m;
                const float new_m = fmaxf(prev_m, tile_max[row]);
                row_m[row] = new_m;
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = lane_y * 4 + i;
            float val = 0.0f;
            if (row < rows_active && col_active) {
                const float new_m = row_m[row];
                const float s = score_tile[row * BC_TILE + col];
                val = __expf(s - new_m);
            }
            prob_tile[row * BC_TILE + col] = val;
            float sum_v = val;
#pragma unroll
            for (int offset = BC_TILE >> 1; offset > 0; offset >>= 1) {
                sum_v += __shfl_down(sum_v, offset, BC_TILE);
            }
            if (lane_x == 0) {
                tile_sum[row] = sum_v;
            }
        }
        __syncthreads();

        if (lane_x == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = lane_y * 4 + i;
                if (row < rows_active) {
                    const float prev_m = row_prev_m[row];
                    const float new_m = row_m[row];
                    const float prev_l = row_l[row];
                    const float sum = tile_sum[row];
                    const float a = (prev_m == NEG_INF) ? 0.0f : __expf(prev_m - new_m);
                    const float weighted_prev = a * prev_l;
                    const float ln = weighted_prev + sum;
                    const float scale = (ln > 0.0f) ? (weighted_prev / ln) : 0.0f;
                    const float inv_l = (ln > 0.0f) ? (1.0f / ln) : 0.0f;
                    row_scale[row] = scale;
                    row_inv_l[row] = inv_l;
                    row_l[row] = ln;
                } else {
                    row_scale[row] = 0.0f;
                    row_inv_l[row] = 0.0f;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = lane_y * 4 + i;
            const float scale = (row < rows_active) ? row_scale[row] : 0.0f;
            o_frag0[i] *= scale;
            o_frag1[i] *= scale;
            o_frag2[i] *= scale;
            o_frag3[i] *= scale;
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = lane_y * 4 + i;
            float val = prob_tile[row * BC_TILE + col];
            if (row < rows_active) {
                val *= row_inv_l[row];
            } else {
                val = 0.0f;
            }
            if (!col_active) {
                val = 0.0f;
            }
            prob_tile[row * BC_TILE + col] = val;
        }
        __syncthreads();

        // Stage V tile (Bc x 64) into shared memory as bf16 (transposed)
        const int total_entries = HEAD_DIM * BC_TILE;
        for (int idx = lane_id; idx < total_entries; idx += 64) {
            const int dim = idx / BC_TILE;
            const int k = idx % BC_TILE;
            v_tile[dim][k] = 0;
        }
        __syncthreads();

        const int total_fills = HEAD_DIM * bc;
        for (int idx = lane_id; idx < total_fills; idx += 64) {
            const int dim = idx / bc;
            const int k = idx % bc;
            const int rt_col = tile_rt[k];
            const long long v_off = static_cast<long long>(rt_col) * step_bytes + head_off_bytes;
            const char* __restrict__ vp_bytes = v_layer_base + v_off;
            float val = 0.0f;
            if constexpr (std::is_same<KV_T, float>::value) {
                val = reinterpret_cast<const float*>(vp_bytes)[dim];
            } else {
                val = bf16_to_f32_intr(reinterpret_cast<const __hip_bfloat16*>(vp_bytes)[dim]);
            }
            v_tile[dim][k] = bf16_bits(__float2bfloat16(val));
        }
        __syncthreads();

        const int k_base = lane_y * 4;
        i16x4 w_frag = {0, 0, 0, 0};
        if (lane_x < rows_active) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int k_idx = k_base + i;
                float val = 0.0f;
                if (k_idx < BC_TILE) {
                    val = prob_tile[lane_x * BC_TILE + k_idx];
                }
                w_frag[i] = bf16_bits(__float2bfloat16(val));
            }
        }

        const int n_col0 = lane_x;
        const int n_col1 = lane_x + 16;
        const int n_col2 = lane_x + 32;
        const int n_col3 = lane_x + 48;

        auto load_v_fragment = [&](int n_col) {
            i16x4 frag = {0, 0, 0, 0};
            if (n_col < HEAD_DIM) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int k_idx = k_base + i;
                    if (k_idx < bc) {
                        frag[i] = v_tile[n_col][k_idx];
                    }
                }
            }
            return frag;
        };

        i16x4 vb0 = load_v_fragment(n_col0);
        i16x4 vb1 = load_v_fragment(n_col1);
        i16x4 vb2 = load_v_fragment(n_col2);
        i16x4 vb3 = load_v_fragment(n_col3);

        o_frag0 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(w_frag, vb0, o_frag0, 0, 0, 0);
        o_frag1 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(w_frag, vb1, o_frag1, 0, 0, 0);
        o_frag2 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(w_frag, vb2, o_frag2, 0, 0, 0);
        o_frag3 = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(w_frag, vb3, o_frag3, 0, 0, 0);
    }

    if (attn_sinks_half != nullptr) {
        if (lane_x == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = lane_y * 4 + i;
                if (row < rows_active) {
                    const int h = h_base + row;
                    const float sink_val = __bfloat162float(attn_sinks_half[h]);
                    if (isfinite(sink_val)) {
                        const float mp = row_m[row];
                        const float lp = row_l[row];
                        const float mn = fmaxf(mp, sink_val);
                        const float a = (mp == NEG_INF) ? 0.0f : __expf(mp - mn);
                        const float b = __expf(sink_val - mn);
                        const float d = a * lp + b;
                        const float scale = (d > 0.0f) ? (a * lp) / d : 1.0f;
                        row_scale[row] = scale;
                        row_m[row] = mn;
                        row_l[row] = d;
                    } else {
                        row_scale[row] = 1.0f;
                    }
                } else {
                    row_scale[row] = 1.0f;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = lane_y * 4 + i;
            const float scale = row_scale[row];
            o_frag0[i] *= scale;
            o_frag1[i] *= scale;
            o_frag2[i] *= scale;
            o_frag3[i] *= scale;
        }
    }

    const int col0 = lane_x;
    const int col1 = lane_x + 16;
    const int col2 = lane_x + 32;
    const int col3 = lane_x + 48;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = lane_y * 4 + i;
        if (row >= rows_active) {
            continue;
        }
        float* __restrict__ o_row = o_block + static_cast<long long>(h_base + row) * HEAD_DIM;
        o_row[col0] = o_frag0[i];
        o_row[col1] = o_frag1[i];
        o_row[col2] = o_frag2[i];
        o_row[col3] = o_frag3[i];
    }
}

void fa(
    const float* q_batch,
    const void*  k_cache,
    const void*  v_cache,
    const float* mask,
    const __hip_bfloat16* attn_sinks,
    float* tb_batch,
    int B,
    int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    int kv_cache_is_fp16,
    const int* d_pos_per_token,
    const int* d_batch_indices,
    long long B_stride,
    int /*max_pos_in_batch*/,
    float* /*workspace_partial_O*/,
    float* /*workspace_partial_m*/,
    float* /*workspace_partial_l*/,
    hipStream_t stream,
    const long long* d_layer_kv_off,
    const int* d_layer_kv_cap,
    const int* d_layer_is_local)
{
    if (B <= 0 || n_attn_heads <= 0 || head_dim != 64) {
        return;
    }

    if (kv_mul > 16 || (n_attn_heads % kv_mul) != 0) {
        // Unsupported configuration for this specialized kernel
        return;
    }

    const __hip_bfloat16* attn_sinks_head =
        (attn_sinks != nullptr) ? (attn_sinks + static_cast<long long>(layer_idx) * n_attn_heads) : nullptr;
    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0) && (mask != nullptr);

    dim3 block(64);
    dim3 grid(n_attn_heads / kv_mul, B);
    const int bc_tile_hint = min(16, seq_len);

    if (kv_cache_is_fp16) {
        if (use_mask) {
            hipLaunchKernelGGL(
                (fa2_decode_mfma_head64<__hip_bfloat16, true>),
                grid, block, 0, stream,
                q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                bc_tile_hint);
        } else {
            hipLaunchKernelGGL(
                (fa2_decode_mfma_head64<__hip_bfloat16, false>),
                grid, block, 0, stream,
                q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                bc_tile_hint);
        }
    } else {
        if (use_mask) {
            hipLaunchKernelGGL(
                (fa2_decode_mfma_head64<float, true>),
                grid, block, 0, stream,
                q_batch, k_cache, v_cache, mask, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                bc_tile_hint);
        } else {
            hipLaunchKernelGGL(
                (fa2_decode_mfma_head64<float, false>),
                grid, block, 0, stream,
                q_batch, k_cache, v_cache, nullptr, attn_sinks_head, tb_batch,
                B, seq_len, head_dim, kv_dim, kv_mul, sliding_window, layer_idx, n_attn_heads,
                d_pos_per_token, d_batch_indices, B_stride,
                d_layer_kv_off, d_layer_kv_cap, d_layer_is_local,
                bc_tile_hint);
        }
    }
    CHECK_HIP(hipGetLastError());
}
