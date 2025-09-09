#include "BLAS.hip"

// ! Vec + Vec
__global__ void vecaddvec_kernel(float* a, float* b, float weight, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        a[i] += b[i] * weight;
}

__global__ void vecaddvec_batch_kernel(float* a_batch, float* b_batch, float weight, int batch_size, int size) {
    int b = blockIdx.y;                          // batch index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // element index

    if (b < batch_size && i < size) {
        int offset = b * size + i;
        a_batch[offset] += b_batch[offset] * weight;
    }
}

void vec_add_vec_gpu(float* a, float* b, float weight, int size) {
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64);
    hipLaunchKernelGGL(vecaddvec_kernel, gridDim, blockDim, 0, 0, a, b, weight, size);

    CHECK_HIP(hipGetLastError());
}

void vec_add_vec_batch_gpu(float* a_batch, float* b_batch, float weight, int batch_size, int size) {
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64, batch_size);
    hipLaunchKernelGGL(vecaddvec_batch_kernel, gridDim, blockDim, 0, 0, a_batch, b_batch, weight, batch_size, size);

    CHECK_HIP(hipGetLastError());
}

// ! Split Gate Up
__global__ void split_gate_up_kernel(float* mlp1_out, float* gate, float* up,
                                     int intermediate_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate_dim) {
        gate[i] = mlp1_out[2 * i];
        up[i] = mlp1_out[2 * i + 1];
    }
}

__global__ void split_gate_up_batch_kernel(float* mlp1_out_batch, float* gate_batch, float* up_batch,
                                           int batch_size, int intermediate_dim) {
    int b = blockIdx.y;                          // batch index
    int i = blockIdx.x * blockDim.x + threadIdx.x; // element index

    if (b < batch_size && i < intermediate_dim) {
        int mlp_offset = b * (2 * intermediate_dim);
        int out_offset = b * intermediate_dim + i;

        gate_batch[out_offset] = mlp1_out_batch[mlp_offset + 2 * i];
        up_batch[out_offset] = mlp1_out_batch[mlp_offset + 2 * i + 1];
    }
}

void split_gate_up_gpu(float* mlp1_out, float* gate, float* up, int intermediate_dim) {
    dim3 blockDim(256);
    dim3 gridDim((intermediate_dim + 255) / 256);
    hipLaunchKernelGGL(split_gate_up_kernel, gridDim, blockDim, 0, 0, mlp1_out, gate, up,
                       intermediate_dim);

    CHECK_HIP(hipGetLastError());
}

void split_gate_up_batch_gpu(float* mlp1_out_batch, float* gate_batch, float* up_batch,
                              int batch_size, int intermediate_dim) {
    dim3 blockDim(256);
    dim3 gridDim((intermediate_dim + 255) / 256, batch_size);
    hipLaunchKernelGGL(split_gate_up_batch_kernel, gridDim, blockDim, 0, 0, mlp1_out_batch, gate_batch, up_batch,
                       batch_size, intermediate_dim);

    CHECK_HIP(hipGetLastError());
}
