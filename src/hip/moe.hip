#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

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
                // vectorized over columns
                const int vecRow = Ktile >> 2;
                for (int j4 = lane; j4 < vecRow; j4 += 64) {
                    const int j = j4 << 2;

                    __hip_bfloat162 g0 = *reinterpret_cast<const __hip_bfloat162*>(&A_half[a_base_gate + k0 + j + 0]);
                    __hip_bfloat162 g1 = *reinterpret_cast<const __hip_bfloat162*>(&A_half[a_base_gate + k0 + j + 2]);
                    __hip_bfloat162 u0 = *reinterpret_cast<const __hip_bfloat162*>(&A_half[a_base_up   + k0 + j + 0]);
                    __hip_bfloat162 u1 = *reinterpret_cast<const __hip_bfloat162*>(&A_half[a_base_up   + k0 + j + 2]);

                    float2 gf0 = __bfloat1622float2(g0);
                    float2 gf1 = __bfloat1622float2(g1);
                    float2 uf0 = __bfloat1622float2(u0);
                    float2 uf1 = __bfloat1622float2(u1);

                    float4 bv = *reinterpret_cast<const float4*>(&sB[j]);

                    acc_gate = fmaf(gf0.x, bv.x, acc_gate);
                    acc_gate = fmaf(gf0.y, bv.y, acc_gate);
                    acc_gate = fmaf(gf1.x, bv.z, acc_gate);
                    acc_gate = fmaf(gf1.y, bv.w, acc_gate);

                    acc_up   = fmaf(uf0.x, bv.x, acc_up);
                    acc_up   = fmaf(uf0.y, bv.y, acc_up);
                    acc_up   = fmaf(uf1.x, bv.z, acc_up);
                    acc_up   = fmaf(uf1.y, bv.w, acc_up);
                }
                for (int j = ((Ktile >> 2) << 2) + lane; j < Ktile; j += 64) {
                    const float b = sB[j];
                    const float ag = __bfloat162float(A_half[a_base_gate + k0 + j]);
                    const float au = __bfloat162float(A_half[a_base_up   + k0 + j]);
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

            gate *= (1.0f / (1.0f + expf(-alpha * gate))); // gate * sigmoid(alpha*gate)
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
