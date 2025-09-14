#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ! MoE SWIGLU Split
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

// ! MoE Scale & Scatter-Add
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
