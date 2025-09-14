#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ! Multi-expert MatVec
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

// ! Multi-expert SWIGLU Split
__global__ void moe_split_swiglu_kernel(
    const int* __restrict__ work_queue,
    int work_start, int work_count,
    const float* __restrict__ mlp1_by_expert,  // [sum Ne, 2I]
    float* __restrict__ gate_by_expert,        // [sum Ne, I] (output)
    int I, float alpha, float swiglu_limit)
{
    const int work_idx = blockIdx.z;
    if (work_idx >= work_count) return;

    const int q   = (work_start + work_idx) * 3;
    const int off = work_queue[q + 1];
    const int Ne  = work_queue[q + 2];

    const int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..I-1
    if (i >= I) return;

    const int grid_rows = gridDim.y;
    const int rows_per_block = (Ne + grid_rows - 1) / grid_rows;
    const int r0 = blockIdx.y * rows_per_block;
    const int rEnd = min(Ne, r0 + rows_per_block);

    const float* __restrict__ src = mlp1_by_expert + (long long)off * (2 * I);
    float*       __restrict__ dst = gate_by_expert + (long long)off * I;

    for (int b = r0; b < rEnd; b++) {
        const int mlp_offset = b * (2 * I);
        const int out_offset = b * I + i;

        float gate_val = src[mlp_offset + 2 * i + 0];
        float up_val   = src[mlp_offset + 2 * i + 1];

        gate_val = fminf(gate_val,  swiglu_limit);
        up_val   = fminf(up_val,    swiglu_limit);
        up_val   = fmaxf(up_val,   -swiglu_limit);

        // SiLU
        gate_val *= (1.0f / (1.0f + expf(-alpha * gate_val)));

        // * (up + 1)
        gate_val *= (up_val + 1.0f);

        dst[out_offset] = gate_val;
    }
}

inline void moe_split_swiglu(
    const int* work_queue, int work_start, int work_count,
    const float* mlp1_by_expert, float* gate_by_expert,
    int I, float alpha, float swiglu_limit, int maxNe, int grid_cap,
    hipStream_t stream)
{
    dim3 block(64);
    const int grid_y = max(1, min(maxNe, grid_cap));
    dim3 grid((I + 63) / 64, grid_y, work_count);
    hipLaunchKernelGGL(moe_split_swiglu_kernel,
        grid, block, 0, stream,
        work_queue, work_start, work_count,
        mlp1_by_expert, gate_by_expert, I, alpha, swiglu_limit);
}

// ! Multi-expert Scale & Scatter-Add
__global__ void moe_scale_scatter_kernel(
    const int* __restrict__ work_queue,
    int work_start, int work_count,
    const float* __restrict__ y_by_expert, // [sum Ne, H]
    const int* __restrict__ tokens_flat,   // [sum Ne]
    const float* __restrict__ weights_flat,// [sum Ne]
    float* __restrict__ e_agg,             // [B, H]
    int H, int B)
{
    const int work_idx = blockIdx.z;
    if (work_idx >= work_count) return;

    const int q   = (work_start + work_idx) * 3;
    const int off = work_queue[q + 1];
    const int Ne  = work_queue[q + 2];

    const int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..H-1
    if (col >= H) return;

    const int grid_rows = gridDim.y;
    const int rows_per_block = (Ne + grid_rows - 1) / grid_rows;
    const int r0 = blockIdx.y * rows_per_block;
    const int rEnd = min(Ne, r0 + rows_per_block);

    const float* __restrict__ Y = y_by_expert + (long long)off * H;
    const int*   __restrict__ t = tokens_flat   + off;
    const float* __restrict__ w = weights_flat  + off;

    for (int row = r0; row < rEnd; row++) {
        const int b = t[row];
        const float val = Y[row * H + col] * w[row];
        atomicAdd(&e_agg[(long long)b * H + col], val);
    }
}

inline void moe_scale_scatter(
    const int* work_queue, int work_start, int work_count,
    const float* y_by_expert, const int* tokens_flat, const float* weights_flat,
    float* e_agg, int H, int batch_size, int maxNe, int grid_cap,
    hipStream_t stream)
{
    dim3 block(256);
    const int grid_y = max(1, min(maxNe, grid_cap));
    dim3 grid((H + 255) / 256, grid_y, work_count);
    hipLaunchKernelGGL(moe_scale_scatter_kernel,
        grid, block, 0, stream,
        work_queue, work_start, work_count,
        y_by_expert, tokens_flat, weights_flat,
        e_agg, H, batch_size);
}
