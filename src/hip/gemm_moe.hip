#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ! MoE MatVec
template <int RB, int KTILE>
__global__ void moe_matvec_kernel(
    const int* __restrict__ work_queue,
    int work_start, int work_count,
    const float* __restrict__ x_by_expert,
    float* __restrict__ out_by_expert,
    const __hip_bfloat16* __restrict__ weights_base,
    const __hip_bfloat16* __restrict__ bias_base,
    int K, int M,
    long long expert_weight_stride,
    long long expert_bias_stride)
{
    const int work_idx = blockIdx.z;
    if (work_idx >= work_count) return;

    const int q   = (work_start + work_idx) * 3;
    const int e   = work_queue[q + 0];
    const int off = work_queue[q + 1];
    const int Ne  = work_queue[q + 2];

    const __hip_bfloat16* __restrict__ A_half = weights_base + (long long)e * expert_weight_stride;
    const __hip_bfloat16* __restrict__ bias   = bias_base ? (bias_base + (long long)e * expert_bias_stride) : nullptr;

    const float* __restrict__ B_batch = x_by_expert + (long long)off * K;
    float*       __restrict__ C_batch = out_by_expert + (long long)off * M;

    const int lane = threadIdx.x;   // 0..63
    const int warp = threadIdx.y;   // 0..RB-1
    const int row  = blockIdx.y * RB + warp;
    const bool valid_row = (row < M);

    const int grid_rows = gridDim.x;
    const int rows_per_block = (Ne + grid_rows - 1) / grid_rows;
    const int r0 = blockIdx.x * rows_per_block;
    const int rEnd = min(Ne, r0 + rows_per_block);

    __shared__ float sB[KTILE] __attribute__((aligned(16)));
    const int threads = blockDim.x * blockDim.y;
    const int tid     = warp * blockDim.x + lane;

    const float bias_row = (bias && valid_row) ? __bfloat162float(bias[row]) : 0.0f;
    const int a_base = valid_row ? (row * K) : 0;

    for (int bidx = r0; bidx < rEnd; bidx++) {
        const float* __restrict__ B = B_batch + (long long)bidx * K;
        float*       __restrict__ C = C_batch + (long long)bidx * M;

        float acc = 0.0f;

        for (int k0 = 0; k0 < K; k0 += KTILE) {
            const int rem   = K - k0;
            const int Ktile = (rem < KTILE) ? rem : KTILE;

            const int vecN = Ktile >> 2;
            for (int j4 = tid; j4 < vecN; j4 += threads) {
                reinterpret_cast<float4*>(sB)[j4] =
                    *reinterpret_cast<const float4*>(&B[k0 + (j4 << 2)]);
            }
            for (int j = (vecN << 2) + tid; j < Ktile; j += threads) {
                sB[j] = B[k0 + j];
            }
            __syncthreads();

            if (valid_row) {
                const int vecRow = Ktile >> 2;
                for (int j4 = lane; j4 < vecRow; j4 += 64) {
                    const int j = j4 << 2;

                    __hip_bfloat162 a0 = *reinterpret_cast<const __hip_bfloat162*>(&A_half[a_base + k0 + j + 0]);
                    __hip_bfloat162 a1 = *reinterpret_cast<const __hip_bfloat162*>(&A_half[a_base + k0 + j + 2]);
                    float2 af0 = __bfloat1622float2(a0);
                    float2 af1 = __bfloat1622float2(a1);

                    float4 bv = *reinterpret_cast<const float4*>(&sB[j]);
                    acc = fmaf(af0.x, bv.x, acc);
                    acc = fmaf(af0.y, bv.y, acc);
                    acc = fmaf(af1.x, bv.z, acc);
                    acc = fmaf(af1.y, bv.w, acc);
                }
                for (int j = ((Ktile >> 2) << 2) + lane; j < Ktile; j += 64) {
                    const float b = sB[j];
                    const float a = __bfloat162float(A_half[a_base + k0 + j]);
                    acc = fmaf(a, b, acc);
                }
            }
            __syncthreads();
        }

        acc = warp_reduce_sum_64(acc);
        if (valid_row && lane == 0) {
            C[row] = acc + bias_row;
        }
    }
}

inline void moe_matvec(
    const int* work_queue, int work_start, int work_count,
    const float* x_by_expert, float* out_by_expert,
    const __hip_bfloat16* weights_base, const __hip_bfloat16* bias_base,
    int K, int M, long long expert_weight_stride, long long expert_bias_stride,
    int maxNe, int grid_cap, hipStream_t stream)
{
    constexpr int RB    = 8;
    constexpr int KTILE = 2048;
    dim3 block(64, RB, 1);
    const int grid_x = max(1, min(maxNe, grid_cap));
    dim3 grid(grid_x, (M + RB - 1) / RB, work_count);
    hipLaunchKernelGGL((moe_matvec_kernel<RB, KTILE>),
        grid, block, 0, stream,
        work_queue, work_start, work_count,
        x_by_expert, out_by_expert,
        weights_base, bias_base,
        K, M, expert_weight_stride, expert_bias_stride);
}
