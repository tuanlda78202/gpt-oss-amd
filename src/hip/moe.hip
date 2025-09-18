#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

static inline __host__ __device__ int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

// ------------------------------------------------------------
// MFMA-backed MoE MLP1 kernel: [Ne, H] x [H, 2I] -> fused SwiGLU [Ne, I]
// ------------------------------------------------------------
template <int MT_M, int MT_N, int BK, int WR, int WC>
__launch_bounds__(64 * WR * WC, 2) __global__ void moe_mlp1_swiglu_mfma_kernel(
    const int* __restrict__ work_queue,
    int work_start,
    int work_count,
    const float* __restrict__ x_by_expert,
    float* __restrict__ gate_by_expert,
    const __hip_bfloat16* __restrict__ weights_base,
    const __hip_bfloat16* __restrict__ bias_base,
    int H,
    int I,
    long long expert_weight_stride,
    long long expert_bias_stride,
    float alpha,
    float swiglu_limit) {
    constexpr int WAVES = WR * WC;
    constexpr int TPB = 64 * WAVES;

    const int work_idx = blockIdx.z;
    if (work_idx >= work_count)
        return;

    const int q = (work_start + work_idx) * 3;
    const int e = work_queue[q + 0];
    const int off = work_queue[q + 1];
    const int Ne = work_queue[q + 2];

    const int M = 2 * I;
    const int N = Ne;
    const int K = H;

    if (M == 0 || N == 0 || K == 0)
        return;

    const __hip_bfloat16* __restrict__ A =
        weights_base + static_cast<long long>(e) * expert_weight_stride;
    const __hip_bfloat16* __restrict__ bias =
        bias_base ? (bias_base + static_cast<long long>(e) * expert_bias_stride) : nullptr;
    const float* __restrict__ B_batch = x_by_expert + static_cast<long long>(off) * H;
    float* __restrict__ O_batch = gate_by_expert + static_cast<long long>(off) * I;

    const int m0 = blockIdx.y * MT_M;
    if (m0 >= M)
        return;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int wave = tid >> 6;
    const int lane = tid & 63;

    const int subtile_r = wave / WC;
    const int subtile_c = wave % WC;

    const int m_base = m0 + subtile_r * 16;

    extern __shared__ char smem_moe[];
    __hip_bfloat16* sA0 = reinterpret_cast<__hip_bfloat16*>(smem_moe);
    __hip_bfloat16* sA1 = sA0 + MT_M * (BK + 1);
    float* sB0 = reinterpret_cast<float*>(sA1 + MT_M * (BK + 1));
    float* sB1 = sB0 + BK * (MT_N + 1);

    using f4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    const int lane_x = lane & 15;
    const int lane_y = lane >> 4;

    auto ldA = [&](int m, int k) -> __hip_bfloat16 {
        return (m >= 0 && m < M && k >= 0 && k < K)
                   ? A[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)]
                   : __float2bfloat16(0.0f);
    };
    auto ldB = [&](int n, int k) -> float {
        return (n >= 0 && n < N && k >= 0 && k < K)
                   ? B_batch[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)]
                   : 0.0f;
    };

    const int elemsA = MT_M * BK;
    const int elemsB = BK * MT_N;
    const int total_tiles_n = (N + MT_N - 1) / MT_N;
    if (total_tiles_n == 0)
        return;

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int tile_x = blockIdx.x; tile_x < total_tiles_n; tile_x += gridDim.x) {
        const int n0 = tile_x * MT_N;
        const int n_base = n0 + subtile_c * 16;

        f4 d = {0.f, 0.f, 0.f, 0.f};

        for (int p = tid; p < elemsA; p += TPB) {
            const int mi = p / BK;
            const int kk = p % BK;
            sA0[mi * (BK + 1) + kk] = ldA(m0 + mi, kk);
        }
        for (int p = tid; p < elemsB; p += TPB) {
            const int ki = p % BK;
            const int nj = p / BK;
            sB0[ki * (MT_N + 1) + nj] = ldB(n0 + nj, ki);
        }
        __syncthreads();

        int ping = 0;

        for (int tile = 0; tile < num_k_tiles; ++tile) {
            __hip_bfloat16* sA = (ping == 0) ? sA0 : sA1;
            float* sB = (ping == 0) ? sB0 : sB1;

            const int next_k0 = (tile + 1) * BK;
            if (tile + 1 < num_k_tiles) {
                __hip_bfloat16* sA_next = (ping == 0) ? sA1 : sA0;
                float* sB_next = (ping == 0) ? sB1 : sB0;

                for (int p = tid; p < elemsA; p += TPB) {
                    const int mi = p / BK;
                    const int kk = p % BK;
                    sA_next[mi * (BK + 1) + kk] = ldA(m0 + mi, next_k0 + kk);
                }
                for (int p = tid; p < elemsB; p += TPB) {
                    const int ki = p % BK;
                    const int nj = p / BK;
                    sB_next[ki * (MT_N + 1) + nj] = ldB(n0 + nj, next_k0 + ki);
                }
            }

#pragma unroll
            for (int kk4 = 0; kk4 < BK; kk4 += 4) {
                const int k_cur = kk4 + lane_y;
                const int a_row = (m_base - m0) + lane_x;
                const int b_col = (n_base - n0) + lane_x;

                float amk = 0.f;
                float bkn = 0.f;
                if (static_cast<unsigned>(a_row) < static_cast<unsigned>(MT_M) &&
                    static_cast<unsigned>(k_cur) < static_cast<unsigned>(BK)) {
                    amk = __bfloat162float(sA[a_row * (BK + 1) + k_cur]);
                }
                if (static_cast<unsigned>(k_cur) < static_cast<unsigned>(BK) &&
                    static_cast<unsigned>(b_col) < static_cast<unsigned>(MT_N)) {
                    bkn = sB[k_cur * (MT_N + 1) + b_col];
                }

                d = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, d, 0, 0, 0);
            }

            __syncthreads();
            ping ^= 1;
        }

        const int n_out = n_base + lane_x;
        if (n_out < N) {
            float* __restrict__ out_row = O_batch + static_cast<size_t>(n_out) * I;

#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int lane_block = lane >> 4; // 0..3
                const int row_local = lane_block * 4 + pair * 2;
                const int gate_row = m_base + row_local;
                const int up_row = gate_row + 1;
                if (gate_row >= M)
                    continue;

                const int i_idx = gate_row >> 1;
                if (i_idx >= I)
                    continue;

                float gate_val = d[pair * 2];
                float up_val = (up_row < M) ? d[pair * 2 + 1] : 0.0f;

                if (bias) {
                    gate_val += __bfloat162float(bias[gate_row]);
                    if (up_row < M) {
                        up_val += __bfloat162float(bias[up_row]);
                    }
                }

                gate_val = fminf(gate_val, swiglu_limit);
                up_val = fminf(up_val, swiglu_limit);
                up_val = fmaxf(up_val, -swiglu_limit);

                const float sigmoid = 1.0f / (1.0f + __expf(-alpha * gate_val));
                float out_val = gate_val * sigmoid * (up_val + 1.0f);

                out_row[i_idx] = out_val;
            }
        }

        __syncthreads();
    }
}

inline void moe_mlp1_swiglu(
    const int* work_queue,
    int work_start,
    int work_count,
    const float* x_by_expert,
    float* gate_by_expert,
    const __hip_bfloat16* weights_base,
    const __hip_bfloat16* bias_base,
    int H,
    int I,
    long long expert_weight_stride,
    long long expert_bias_stride,
    float alpha,
    float swiglu_limit,
    int maxNe,
    int grid_cap,
    hipStream_t stream) {
    constexpr int MT_M = 64;
    constexpr int MT_N = 32;
    constexpr int BK = 32;
    constexpr int WR = 4;
    constexpr int WC = 2;
    constexpr int WAVES = WR * WC;

    const int grid_x = max(1, min(grid_cap, ceil_div_int(maxNe, MT_N)));
    const int grid_y = max(1, ceil_div_int(2 * I, MT_M));

    dim3 block(64 * WAVES, 1, 1);
    dim3 grid(grid_x, grid_y, work_count);

    const size_t shmem_bytes = 2 * (MT_M * (BK + 1) * sizeof(__hip_bfloat16)) +
                               2 * (BK * (MT_N + 1) * sizeof(float));

    hipLaunchKernelGGL((moe_mlp1_swiglu_mfma_kernel<MT_M, MT_N, BK, WR, WC>), grid, block,
                       shmem_bytes, stream, work_queue, work_start, work_count, x_by_expert,
                       gate_by_expert, weights_base, bias_base, H, I, expert_weight_stride,
                       expert_bias_stride, alpha, swiglu_limit);
}

// ------------------------------------------------------------
// MFMA-backed MoE MLP2 kernel: [Ne, I] x [I, H] -> scatter to e_agg
// ------------------------------------------------------------
template <int MT_M, int MT_N, int BK, int WR, int WC>
__launch_bounds__(64 * WR * WC, 2) __global__ void moe_mlp2_scatter_mfma_kernel(
    const int* __restrict__ work_queue,
    int work_start,
    int work_count,
    const float* __restrict__ gate_by_expert,
    const int* __restrict__ tokens_flat,
    const float* __restrict__ weights_flat,
    float* __restrict__ e_agg,
    const __hip_bfloat16* __restrict__ weights_base,
    const __hip_bfloat16* __restrict__ bias_base,
    int I,
    int H,
    long long expert_weight_stride,
    long long expert_bias_stride) {
    constexpr int WAVES = WR * WC;
    constexpr int TPB = 64 * WAVES;

    const int work_idx = blockIdx.z;
    if (work_idx >= work_count)
        return;

    const int q = (work_start + work_idx) * 3;
    const int e = work_queue[q + 0];
    const int off = work_queue[q + 1];
    const int Ne = work_queue[q + 2];

    const int M = H;
    const int N = Ne;
    const int K = I;

    if (M == 0 || N == 0 || K == 0)
        return;

    const __hip_bfloat16* __restrict__ A =
        weights_base + static_cast<long long>(e) * expert_weight_stride;
    const __hip_bfloat16* __restrict__ bias =
        bias_base ? (bias_base + static_cast<long long>(e) * expert_bias_stride) : nullptr;
    const float* __restrict__ B_batch = gate_by_expert + static_cast<long long>(off) * I;
    const int* __restrict__ tokens = tokens_flat + off;
    const float* __restrict__ scales = weights_flat + off;

    const int m0 = blockIdx.y * MT_M;
    if (m0 >= M)
        return;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int wave = tid >> 6;
    const int lane = tid & 63;

    const int subtile_r = wave / WC;
    const int subtile_c = wave % WC;

    const int m_base = m0 + subtile_r * 16;

    extern __shared__ char smem_moe[];
    __hip_bfloat16* sA0 = reinterpret_cast<__hip_bfloat16*>(smem_moe);
    __hip_bfloat16* sA1 = sA0 + MT_M * (BK + 1);
    float* sB0 = reinterpret_cast<float*>(sA1 + MT_M * (BK + 1));
    float* sB1 = sB0 + BK * (MT_N + 1);

    using f4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

    const int lane_x = lane & 15;
    const int lane_y = lane >> 4;

    auto ldA = [&](int m, int k) -> __hip_bfloat16 {
        return (m >= 0 && m < M && k >= 0 && k < K)
                   ? A[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)]
                   : __float2bfloat16(0.0f);
    };
    auto ldB = [&](int n, int k) -> float {
        return (n >= 0 && n < N && k >= 0 && k < K)
                   ? B_batch[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)]
                   : 0.0f;
    };

    const int elemsA = MT_M * BK;
    const int elemsB = BK * MT_N;
    const int total_tiles_n = (N + MT_N - 1) / MT_N;
    if (total_tiles_n == 0)
        return;

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int tile_x = blockIdx.x; tile_x < total_tiles_n; tile_x += gridDim.x) {
        const int n0 = tile_x * MT_N;
        const int n_base = n0 + subtile_c * 16;

        f4 d = {0.f, 0.f, 0.f, 0.f};

        for (int p = tid; p < elemsA; p += TPB) {
            const int mi = p / BK;
            const int kk = p % BK;
            sA0[mi * (BK + 1) + kk] = ldA(m0 + mi, kk);
        }
        for (int p = tid; p < elemsB; p += TPB) {
            const int ki = p % BK;
            const int nj = p / BK;
            sB0[ki * (MT_N + 1) + nj] = ldB(n0 + nj, ki);
        }
        __syncthreads();

        int ping = 0;

        for (int tile = 0; tile < num_k_tiles; ++tile) {
            __hip_bfloat16* sA = (ping == 0) ? sA0 : sA1;
            float* sB = (ping == 0) ? sB0 : sB1;

            const int next_k0 = (tile + 1) * BK;
            if (tile + 1 < num_k_tiles) {
                __hip_bfloat16* sA_next = (ping == 0) ? sA1 : sA0;
                float* sB_next = (ping == 0) ? sB1 : sB0;

                for (int p = tid; p < elemsA; p += TPB) {
                    const int mi = p / BK;
                    const int kk = p % BK;
                    sA_next[mi * (BK + 1) + kk] = ldA(m0 + mi, next_k0 + kk);
                }
                for (int p = tid; p < elemsB; p += TPB) {
                    const int ki = p % BK;
                    const int nj = p / BK;
                    sB_next[ki * (MT_N + 1) + nj] = ldB(n0 + nj, next_k0 + ki);
                }
            }

#pragma unroll
            for (int kk4 = 0; kk4 < BK; kk4 += 4) {
                const int k_cur = kk4 + lane_y;
                const int a_row = (m_base - m0) + lane_x;
                const int b_col = (n_base - n0) + lane_x;

                float amk = 0.f;
                float bkn = 0.f;
                if (static_cast<unsigned>(a_row) < static_cast<unsigned>(MT_M) &&
                    static_cast<unsigned>(k_cur) < static_cast<unsigned>(BK)) {
                    amk = __bfloat162float(sA[a_row * (BK + 1) + k_cur]);
                }
                if (static_cast<unsigned>(k_cur) < static_cast<unsigned>(BK) &&
                    static_cast<unsigned>(b_col) < static_cast<unsigned>(MT_N)) {
                    bkn = sB[k_cur * (MT_N + 1) + b_col];
                }

                d = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, d, 0, 0, 0);
            }

            __syncthreads();
            ping ^= 1;
        }

        const int n_out = n_base + lane_x;
        if (n_out < N) {
            const int token = tokens[n_out];
            const float scale = scales[n_out];
            float* __restrict__ eagg_row = e_agg + static_cast<size_t>(token) * H;

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int lane_block = lane >> 4; // 0..3
                const int row_local = lane_block * 4 + i;
                const int m_out = m_base + row_local;
                if (m_out >= M)
                    continue;

                float out_val = d[i];
                if (bias) {
                    out_val += __bfloat162float(bias[m_out]);
                }
                atomicAdd(&eagg_row[m_out], out_val * scale);
            }
        }

        __syncthreads();
    }
}

inline void moe_mlp2_scatter(
    const int* work_queue,
    int work_start,
    int work_count,
    const float* gate_by_expert,
    const int* tokens_flat,
    const float* weights_flat,
    float* e_agg,
    const __hip_bfloat16* weights_base,
    const __hip_bfloat16* bias_base,
    int I,
    int H,
    long long expert_weight_stride,
    long long expert_bias_stride,
    int maxNe,
    int grid_cap,
    hipStream_t stream) {
    constexpr int MT_M = 64;
    constexpr int MT_N = 32;
    constexpr int BK = 32;
    constexpr int WR = 4;
    constexpr int WC = 2;
    constexpr int WAVES = WR * WC;

    const int grid_x = max(1, min(grid_cap, ceil_div_int(maxNe, MT_N)));
    const int grid_y = max(1, ceil_div_int(H, MT_M));

    dim3 block(64 * WAVES, 1, 1);
    dim3 grid(grid_x, grid_y, work_count);

    const size_t shmem_bytes = 2 * (MT_M * (BK + 1) * sizeof(__hip_bfloat16)) +
                               2 * (BK * (MT_N + 1) * sizeof(float));

    hipLaunchKernelGGL((moe_mlp2_scatter_mfma_kernel<MT_M, MT_N, BK, WR, WC>), grid, block,
                       shmem_bytes, stream, work_queue, work_start, work_count, gate_by_expert,
                       tokens_flat, weights_flat, e_agg, weights_base, bias_base, I, H,
                       expert_weight_stride, expert_bias_stride);
}
