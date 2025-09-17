#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

__device__ __forceinline__ float4 bf16x4_to_float4(const __hip_bfloat16* ptr) {
    float4 result;
    const __hip_bfloat162 lo = *reinterpret_cast<const __hip_bfloat162*>(ptr + 0);
    const __hip_bfloat162 hi = *reinterpret_cast<const __hip_bfloat162*>(ptr + 2);
    const float2 flo = __bfloat1622float2(lo);
    const float2 fhi = __bfloat1622float2(hi);
    result.x = flo.x;
    result.y = flo.y;
    result.z = fhi.x;
    result.w = fhi.y;
    return result;
}

__device__ __forceinline__ float fma_dot4(const float4& a, const float4& b, float acc) {
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
    acc = fmaf(a.z, b.z, acc);
    return fmaf(a.w, b.w, acc);
}

// MLP1 (GEMV) + SwiGLU: [Ne,H] x [H,2I] -> [Ne,I]
template <int RB, int KTILE>
__global__ void moe_mlp1_swiglu_kernel(
    const int* __restrict__ work_queue,
    int work_start, int work_count,
    const float* __restrict__ x_by_expert,    // [sumNe, H]
    float* __restrict__ gate_by_expert,       // [sumNe, I] (output)
    const __hip_bfloat16* __restrict__ weights_base, // [E][2I,H] BF16
    const __hip_bfloat16* __restrict__ bias_base,    // [E][2I] BF16 (optional)
    int H, int I,
    long long expert_weight_stride,           // stride in elements for [2I,H]
    long long expert_bias_stride,             // stride in elements for [2I]
    float alpha, float swiglu_limit)
{
    const int work_idx = blockIdx.z;
    if (work_idx >= work_count) return;

    const int q   = (work_start + work_idx) * 3;
    const int e   = work_queue[q + 0];
    const int off = work_queue[q + 1];
    const int Ne  = work_queue[q + 2];

    const __hip_bfloat16* __restrict__ A_half = weights_base + (long long)e * expert_weight_stride; // rows=2I, cols=H
    const __hip_bfloat16* __restrict__ bias   = bias_base ? (bias_base + (long long)e * expert_bias_stride) : nullptr;

    const float* __restrict__ B_batch = x_by_expert    + (long long)off * H; // [Ne,H]
    float*       __restrict__ O_batch = gate_by_expert + (long long)off * I; // [Ne,I]

    const int lane = threadIdx.x;   // 0..63
    const int warp = threadIdx.y;   // 0..RB-1
    const int irow = blockIdx.y * RB + warp;   // 0..I-1
    const bool valid_row = (irow < I);

    const int grid_rows     = gridDim.x;
    const int rows_per_block= (Ne + grid_rows - 1) / grid_rows;
    const int r0   = blockIdx.x * rows_per_block;
    const int rEnd = min(Ne, r0 + rows_per_block);

    __shared__ float sB[KTILE] __attribute__((aligned(16)));
    const int threads = blockDim.x * blockDim.y;
    const int tid     = warp * blockDim.x + lane;

    const int gate_row = (irow << 1);
    const int up_row   = gate_row + 1;
    const long long a_base_gate = valid_row ? ((long long)gate_row * H) : 0;
    const long long a_base_up   = valid_row ? ((long long)up_row   * H) : 0;

    const float bias_gate = (bias && valid_row) ? __bfloat162float(bias[gate_row]) : 0.0f;
    const float bias_up   = (bias && valid_row) ? __bfloat162float(bias[up_row])   : 0.0f;

    for (int bidx = r0; bidx < rEnd; ++bidx) {
        const float* __restrict__ B = B_batch + (long long)bidx * H;
        float*       __restrict__ O = O_batch + (long long)bidx * I;

        float acc_gate = 0.0f;
        float acc_up   = 0.0f;

        for (int k0 = 0; k0 < H; k0 += KTILE) {
            const int rem   = H - k0;
            const int Ktile = (rem < KTILE) ? rem : KTILE;

            // stage B tile -> sB
            const float* __restrict__ B_tile = B + k0;
            const int vecN = Ktile >> 2;
            float4* __restrict__ sB_vec = reinterpret_cast<float4*>(sB);
            for (int j4 = tid; j4 < vecN; j4 += threads) {
                const int offset = j4 << 2;
                sB_vec[j4] = *reinterpret_cast<const float4*>(&B_tile[offset]);
            }
            for (int j = (vecN << 2) + tid; j < Ktile; j += threads) {
                sB[j] = B_tile[j];
            }
            __syncthreads();

            if (valid_row) {
                // vectorized over columns
                const int vecRow = Ktile >> 2;
                const int warp_width = 64;
                const int vec_stride = warp_width << 2;

                const __hip_bfloat16* __restrict__ gate_tile = A_half + a_base_gate + k0;
                const __hip_bfloat16* __restrict__ up_tile   = A_half + a_base_up   + k0;
                const float4* __restrict__ sB_vec_const = reinterpret_cast<const float4*>(sB);

                for (int j4 = lane, offset = (lane << 2); j4 < vecRow; j4 += warp_width, offset += vec_stride) {
                    const float4 b4 = sB_vec_const[j4];
                    const float4 g4 = bf16x4_to_float4(gate_tile + offset);
                    const float4 u4 = bf16x4_to_float4(up_tile   + offset);

                    acc_gate = fma_dot4(g4, b4, acc_gate);
                    acc_up   = fma_dot4(u4, b4, acc_up);
                }

                const int scalar_start = (vecRow << 2) + lane;
                for (int j = scalar_start; j < Ktile; j += warp_width) {
                    const float b  = sB[j];
                    const float ag = __bfloat162float(gate_tile[j]);
                    const float au = __bfloat162float(up_tile[j]);
                    acc_gate = fmaf(ag, b, acc_gate);
                    acc_up   = fmaf(au, b, acc_up);
                }
            }
            __syncthreads();
        }

        // reduce within 64-lane warp
        acc_gate = warp_reduce_sum_64(acc_gate);
        acc_up   = warp_reduce_sum_64(acc_up);

        if (valid_row && lane == 0) {
            float gate = acc_gate + bias_gate;
            float up   = acc_up   + bias_up;

            gate = fminf(gate,  swiglu_limit);
            up   = fminf(up,    swiglu_limit);
            up   = fmaxf(up,   -swiglu_limit);

            gate *= (1.0f / (1.0f + __expf(-alpha * gate))); // gate * sigmoid(alpha*gate)
            gate *= (up + 1.0f);

            O[irow] = gate;
        }
    }
}

inline void moe_mlp1_swiglu(
    const int* work_queue, int work_start, int work_count,
    const float* x_by_expert, float* gate_by_expert,
    const __hip_bfloat16* weights_base, const __hip_bfloat16* bias_base,
    int H, int I, long long expert_weight_stride, long long expert_bias_stride,
    float alpha, float swiglu_limit, int maxNe, int grid_cap, hipStream_t stream)
{
    constexpr int RB    = 8;
    constexpr int KTILE = 2048;
    dim3 block(64, RB, 1);
    const int grid_x = max(1, min(maxNe, grid_cap));
    dim3 grid(grid_x, (I + RB - 1) / RB, work_count);
    hipLaunchKernelGGL((moe_mlp1_swiglu_kernel<RB, KTILE>),
        grid, block, 0, stream,
        work_queue, work_start, work_count,
        x_by_expert, gate_by_expert,
        weights_base, bias_base,
        H, I, expert_weight_stride, expert_bias_stride,
        alpha, swiglu_limit);
}

// ! MLP2 (GEMV) + Scale&Scatter:   [Ne,I] x [I,H] -> e_agg[batch,H]
template <int RB, int KTILE>
__global__ void moe_mlp2_scatter_kernel(
    const int* __restrict__ work_queue,
    int work_start, int work_count,
    const float* __restrict__ gate_by_expert, // [sumNe, I]
    const int*   __restrict__ tokens_flat,    // [sumNe]
    const float* __restrict__ weights_flat,   // [sumNe]
    float* __restrict__ e_agg,                // [B, H]
    const __hip_bfloat16* __restrict__ weights_base, // [E][H,I] BF16  (rows=H, cols=I)
    const __hip_bfloat16* __restrict__ bias_base,    // [E][H]   BF16  (optional)
    int I, int H, long long expert_weight_stride, long long expert_bias_stride)
{
    const int work_idx = blockIdx.z;
    if (work_idx >= work_count) return;

    const int q   = (work_start + work_idx) * 3;
    const int e   = work_queue[q + 0];
    const int off = work_queue[q + 1];
    const int Ne  = work_queue[q + 2];

    const __hip_bfloat16* __restrict__ A_half = weights_base + (long long)e * expert_weight_stride; // rows=H, cols=I
    const __hip_bfloat16* __restrict__ bias   = bias_base ? (bias_base + (long long)e * expert_bias_stride) : nullptr;

    const float* __restrict__ B_batch = gate_by_expert + (long long)off * I; // [Ne,I]

    const int lane = threadIdx.x;   // 0..63
    const int warp = threadIdx.y;   // 0..RB-1
    const int rowH = blockIdx.y * RB + warp;  // output row 0..H-1
    const bool valid_row = (rowH < H);

    const int grid_rows     = gridDim.x;
    const int rows_per_block= (Ne + grid_rows - 1) / grid_rows;
    const int r0   = blockIdx.x * rows_per_block;
    const int rEnd = min(Ne, r0 + rows_per_block);

    __shared__ float sB[KTILE] __attribute__((aligned(16)));
    const int threads = blockDim.x * blockDim.y;
    const int tid     = warp * blockDim.x + lane;

    const float bias_row = (bias && valid_row) ? __bfloat162float(bias[rowH]) : 0.0f;
    const long long a_base = valid_row ? ((long long)rowH * I) : 0;

    for (int bidx = r0; bidx < rEnd; ++bidx) {
        const float* __restrict__ B = B_batch + (long long)bidx * I;

        float acc = 0.0f;

        // GEMV: (rowH,:) dot B
        for (int k0 = 0; k0 < I; k0 += KTILE) {
            const int rem   = I - k0;
            const int Ktile = (rem < KTILE) ? rem : KTILE;

            const float* __restrict__ B_tile = B + k0;
            const int vecN = Ktile >> 2;
            float4* __restrict__ sB_vec = reinterpret_cast<float4*>(sB);
            for (int j4 = tid; j4 < vecN; j4 += threads) {
                const int offset = j4 << 2;
                sB_vec[j4] = *reinterpret_cast<const float4*>(&B_tile[offset]);
            }
            for (int j = (vecN << 2) + tid; j < Ktile; j += threads) {
                sB[j] = B_tile[j];
            }
            __syncthreads();

            if (valid_row) {
                const int vecRow = Ktile >> 2;
                const int warp_width = 64;
                const int vec_stride = warp_width << 2;

                const __hip_bfloat16* __restrict__ row_tile = A_half + a_base + k0;
                const float4* __restrict__ sB_vec_const = reinterpret_cast<const float4*>(sB);

                for (int j4 = lane, offset = (lane << 2); j4 < vecRow; j4 += warp_width, offset += vec_stride) {
                    const float4 b4 = sB_vec_const[j4];
                    const float4 a4 = bf16x4_to_float4(row_tile + offset);
                    acc = fma_dot4(a4, b4, acc);
                }

                const int scalar_start = (vecRow << 2) + lane;
                for (int j = scalar_start; j < Ktile; j += warp_width) {
                    const float b = sB[j];
                    const float a = __bfloat162float(row_tile[j]);
                    acc = fmaf(a, b, acc);
                }
            }
            __syncthreads();
        }

        acc = warp_reduce_sum_64(acc);

        if (valid_row && lane == 0) {
            const int  b = tokens_flat [off + bidx];
            const float w = weights_flat[off + bidx];
            const float val = (acc + bias_row) * w;
            atomicAdd(&e_agg[(long long)b * H + rowH], val);
        }
    }
}

inline void moe_mlp2_scatter(
    const int* work_queue, int work_start, int work_count,
    const float* gate_by_expert, const int* tokens_flat, const float* weights_flat,
    float* e_agg,
    const __hip_bfloat16* weights_base, const __hip_bfloat16* bias_base,
    int I, int H, long long expert_weight_stride, long long expert_bias_stride,
    int maxNe, int grid_cap, hipStream_t stream)
{
    constexpr int RB    = 8;
    constexpr int KTILE = 2048;
    dim3 block(64, RB, 1);
    const int grid_x = max(1, min(maxNe, grid_cap));
    dim3 grid(grid_x, (H + RB - 1) / RB, work_count);
    hipLaunchKernelGGL((moe_mlp2_scatter_kernel<RB, KTILE>),
        grid, block, 0, stream,
        work_queue, work_start, work_count,
        gate_by_expert, tokens_flat, weights_flat, e_agg,
        weights_base, bias_base,
        I, H, expert_weight_stride, expert_bias_stride);
}
