// ! Vec + Vec
__global__ void vecadd_kernel(float* a_batch, float* b_batch, float weight, int batch_size, int size) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < batch_size && i < size) {
        int offset = b * size + i;
        a_batch[offset] += b_batch[offset] * weight;
    }
}

void vecadd(float* a_batch, float* b_batch, float weight, int batch_size, int size) {
    dim3 blockDim(64);
    dim3 gridDim((size + 64 - 1) / 64, batch_size);
    hipLaunchKernelGGL(vecadd_kernel, gridDim, blockDim, 0, 0, a_batch, b_batch, weight, batch_size, size);
    CHECK_HIP(hipGetLastError());
}
