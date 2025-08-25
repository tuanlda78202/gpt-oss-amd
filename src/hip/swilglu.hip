#include "../BLAS.hip"

// '_s_' = single precision (float)
__global__ void thaDNN_s_swiglu_kernel(float* hb, float* hb2, int hidden_dim, float alpha,
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

// '_s_' = single precision
// input: hb, hb2 allocated on device
void thaDNN_s_swiglu(float* hb, float* hb2, int hidden_dim, float alpha, float swiglu_limit) {
    if (hidden_dim == 0 || hb == nullptr || hb2 == nullptr) {
        printf("THABLAS SwiGLU ERROR: INVALID ARGUMENT\n");
        fflush(stdout);
        return;
    }

    dim3 blockDim(64);
    dim3 gridDim((hidden_dim + blockDim.x - 1) / blockDim.x);
    hipLaunchKernelGGL(thaDNN_s_swiglu_kernel, gridDim, blockDim, 0, 0, hb, hb2, hidden_dim, alpha,
                       swiglu_limit);

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

void swiglu_hip(float* gate, float* up, float* gate_up, int intermediate_dim, float alpha,
                float swiglu_limit) {
    float *gate_d, *up_d, *gate_up_d;
    CHECK_HIP(hipMalloc(&gate_d, intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&up_d, intermediate_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&gate_up_d, intermediate_dim * sizeof(float)));

    CHECK_HIP(hipMemcpy(gate_d, gate, intermediate_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(up_d, up, intermediate_dim * sizeof(float), hipMemcpyHostToDevice));

    thaDNN_s_swiglu(gate_d, up_d, intermediate_dim, alpha, swiglu_limit);

    CHECK_HIP(hipMemcpy(gate_up, gate_d, intermediate_dim * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipFree(gate_d));
    CHECK_HIP(hipFree(up_d));
    CHECK_HIP(hipFree(gate_up_d));
}
