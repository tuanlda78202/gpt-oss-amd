#include "BLAS.hip"

__global__ void swiglu_gpu_kernel(float* hb, float* hb2, int hidden_dim, float alpha,
                                       float swiglu_limit) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= hidden_dim)
        return;

    // Get values with clamping
    float val = hb[i];
    float up_val = hb2[i];

    // Clamping
    if (val > swiglu_limit)
        val = swiglu_limit;
    if (up_val > swiglu_limit)
        up_val = swiglu_limit;
    if (up_val < -swiglu_limit)
        up_val = -swiglu_limit;

    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-alpha * val)));

    // elementwise multiply with up(x) + 1.0f (gpt-oss adds an extra bias of 1 to the up layer)
    val *= (up_val + 1.0f);
    hb[i] = val;
}

void swiglu_gpu(float* hb, float* hb2, int hidden_dim, float alpha, float swiglu_limit) {
    if (hidden_dim == 0 || hb == nullptr || hb2 == nullptr) {
        printf("THABLAS SwiGLU ERROR: INVALID ARGUMENT\n");
        fflush(stdout);
        return;
    }

    dim3 blockDim(64);
    dim3 gridDim((hidden_dim + blockDim.x - 1) / blockDim.x);
    hipLaunchKernelGGL(swiglu_gpu_kernel, gridDim, blockDim, 0, 0, hb, hb2, hidden_dim, alpha,
                       swiglu_limit);

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}
