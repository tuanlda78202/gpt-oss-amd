#include "BLAS.hip"
#include <cmath>

// ====== SOFTMAX KERNEL (copied from softmax.hip) ======
__global__ void attention_softmax_kernel(float* x, int size) {
    int lx = threadIdx.x;
    int bDim = blockDim.x;

    float private_max_val = -3.402e+38;
    __shared__ float max_val;
    for (int i = lx; i < size; i += bDim) {
        private_max_val = std::max(private_max_val, x[i]);
    }

    private_max_val = block_reduce_max(private_max_val);
    if (lx == 0) {
        max_val = private_max_val;
    }
    __syncthreads();
    private_max_val = max_val;

    float private_sum = 0.0f, tmp;
    __shared__ float sum;
    for (int i = lx; i < size; i += bDim) {
        tmp = expf(x[i] - private_max_val);
        x[i] = tmp;
        private_sum += tmp;
    }

    private_sum = block_reduce_sum(private_sum);
    if (lx == 0) {
        sum = private_sum;
    }
    __syncthreads();
    private_sum = sum;

    for (int i = lx; i < size; i += bDim) {
        x[i] /= private_sum;
    }
}

// ====== ATTENTION SCORE COMPUTATION KERNEL ======
__global__ void compute_attention_scores_kernel(float* q, float* k_cache, float* att_scores,
                                                float* mask, int pos, int seq_len, int head_dim,
                                                int kv_dim, int kv_mul, int sliding_window,
                                                int layer_idx, int n_attn_heads) {

    int h = blockIdx.x; // head index
    int t = blockIdx.y; // timestep index
    int tid = threadIdx.x;

    if (h >= n_attn_heads || t > pos)
        return;

    // Get the query vector for this head
    float* q_head = q + h * head_dim;

    // Get the key vector for this head and timestep (GQA)
    float* k_head = k_cache + t * kv_dim + (h / kv_mul) * head_dim;

    // Compute dot product using parallel reduction
    __shared__ float shared_sum[256];
    float local_sum = 0.0f;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        local_sum += q_head[i] * k_head[i];
    }

    // Reduce within block
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        float score = shared_sum[0] / sqrtf((float)head_dim);

        // Apply sliding window mask if enabled
        if (sliding_window > 0 && (layer_idx % 2 == 0)) {
            score += mask[pos * seq_len + t];
        }

        att_scores[h * seq_len + t] = score;
    }
}

// ====== ATTENTION SINK SCORE KERNEL ======
__global__ void add_attention_sink_kernel(float* att_scores, float* attn_sinks, int pos,
                                          int seq_len, int n_attn_heads, int layer_idx) {
    int h = blockIdx.x;
    if (h >= n_attn_heads)
        return;

    att_scores[h * seq_len + pos + 1] = attn_sinks[h];
}

// ====== WEIGHTED VALUE ACCUMULATION KERNEL ======
__global__ void weighted_value_accumulation_kernel(float* att_scores, float* v_cache, float* tb,
                                                   int pos, int seq_len, int head_dim, int kv_dim,
                                                   int kv_mul, int n_attn_heads) {

    int h = blockIdx.x;                            // head index
    int i = blockIdx.y * blockDim.x + threadIdx.x; // dimension index

    if (h >= n_attn_heads || i >= head_dim)
        return;

    float* tb_head = tb + h * head_dim;
    float* att_head = att_scores + h * seq_len;

    float weighted_sum = 0.0f;

    // Accumulate weighted values across all timesteps
    for (int t = 0; t <= pos; t++) {
        float* v_head = v_cache + t * kv_dim + (h / kv_mul) * head_dim;
        float att_weight = att_head[t];
        weighted_sum += att_weight * v_head[i];
    }

    tb_head[i] = weighted_sum;
}

// ====== INDIVIDUAL STEP HOST FUNCTIONS ======

void compute_attention_scores_hip(float* q, float* k_cache, float* att_scores, float* mask, int pos,
                                  int seq_len, int head_dim, int kv_dim, int kv_mul,
                                  int sliding_window, int layer_idx, int n_attn_heads) {

    if (n_attn_heads <= 0 || head_dim <= 0 || pos < 0 || seq_len <= 0) {
        printf("ATTN SCORES ERROR: INVALID ARGUMENT\n");
        fflush(stdout);
        return;
    }

    float *q_d, *k_cache_d, *att_scores_d;
    float* mask_d = nullptr;
    CHECK_HIP(hipMalloc(&q_d, n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&k_cache_d, (pos + 1) * kv_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&att_scores_d, n_attn_heads * seq_len * sizeof(float)));
    if (mask != nullptr) {
        CHECK_HIP(hipMalloc(&mask_d, seq_len * seq_len * sizeof(float)));
        CHECK_HIP(
            hipMemcpy(mask_d, mask, seq_len * seq_len * sizeof(float), hipMemcpyHostToDevice));
    }

    CHECK_HIP(hipMemcpy(q_d, q, n_attn_heads * head_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(k_cache_d, k_cache, (pos + 1) * kv_dim * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(att_scores_d, 0, n_attn_heads * seq_len * sizeof(float)));

    dim3 block_dim_attn(256);
    dim3 grid_dim_attn(n_attn_heads, pos + 1);
    hipLaunchKernelGGL(compute_attention_scores_kernel, grid_dim_attn, block_dim_attn, 0, 0, q_d,
                       k_cache_d, att_scores_d, mask_d, pos, seq_len, head_dim, kv_dim, kv_mul,
                       sliding_window, layer_idx, n_attn_heads);

    CHECK_HIP(hipMemcpy(att_scores, att_scores_d, n_attn_heads * seq_len * sizeof(float),
                        hipMemcpyDeviceToHost));

    CHECK_HIP(hipFree(q_d));
    CHECK_HIP(hipFree(k_cache_d));
    CHECK_HIP(hipFree(att_scores_d));
    if (mask_d != nullptr)
        CHECK_HIP(hipFree(mask_d));

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

void add_attention_sink_hip(float* att_scores, float* attn_sinks, int pos, int seq_len,
                            int n_attn_heads, int layer_idx) {
    if (n_attn_heads <= 0 || pos < 0 || seq_len <= 0) {
        printf("ATTN SINK ERROR: INVALID ARGUMENT\n");
        fflush(stdout);
        return;
    }

    float *att_scores_d, *attn_sinks_d;
    CHECK_HIP(hipMalloc(&att_scores_d, n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&attn_sinks_d, n_attn_heads * sizeof(float)));
    CHECK_HIP(hipMemcpy(att_scores_d, att_scores, n_attn_heads * seq_len * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(attn_sinks_d, attn_sinks, n_attn_heads * sizeof(float), hipMemcpyHostToDevice));

    dim3 block_dim_sink(1);
    dim3 grid_dim_sink(n_attn_heads);
    hipLaunchKernelGGL(add_attention_sink_kernel, grid_dim_sink, block_dim_sink, 0, 0, att_scores_d,
                       attn_sinks_d, pos, seq_len, n_attn_heads, layer_idx);

    CHECK_HIP(hipMemcpy(att_scores, att_scores_d, n_attn_heads * seq_len * sizeof(float),
                        hipMemcpyDeviceToHost));

    CHECK_HIP(hipFree(att_scores_d));
    CHECK_HIP(hipFree(attn_sinks_d));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

void softmax_attention_hip(float* att_scores, int pos, int seq_len, int n_attn_heads) {
    if (n_attn_heads <= 0 || pos < 0 || seq_len <= 0) {
        printf("ATTN SOFTMAX ERROR: INVALID ARGUMENT\n");
        fflush(stdout);
        return;
    }

    float* att_scores_d;
    CHECK_HIP(hipMalloc(&att_scores_d, n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMemcpy(att_scores_d, att_scores, n_attn_heads * seq_len * sizeof(float),
                        hipMemcpyHostToDevice));

    for (int h = 0; h < n_attn_heads; h++) {
        float* att_head_d = att_scores_d + h * seq_len;
        dim3 blockDim(1024);
        dim3 gridDim(1);
        hipLaunchKernelGGL(attention_softmax_kernel, gridDim, blockDim, 0, 0, att_head_d, pos + 2);
    }

    CHECK_HIP(hipMemcpy(att_scores, att_scores_d, n_attn_heads * seq_len * sizeof(float),
                        hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(att_scores_d));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

void weighted_value_accumulation_hip(float* att_scores, float* v_cache, float* tb, int pos,
                                     int seq_len, int head_dim, int kv_dim, int kv_mul,
                                     int n_attn_heads) {

    if (n_attn_heads <= 0 || head_dim <= 0 || pos < 0 || seq_len <= 0) {
        printf("ATTN ACCUM ERROR: INVALID ARGUMENT\n");
        fflush(stdout);
        return;
    }

    float *att_scores_d, *v_cache_d, *tb_d;
    CHECK_HIP(hipMalloc(&att_scores_d, n_attn_heads * seq_len * sizeof(float)));
    CHECK_HIP(hipMalloc(&v_cache_d, (pos + 1) * kv_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&tb_d, n_attn_heads * head_dim * sizeof(float)));
    CHECK_HIP(hipMemcpy(att_scores_d, att_scores, n_attn_heads * seq_len * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(
        hipMemcpy(v_cache_d, v_cache, (pos + 1) * kv_dim * sizeof(float), hipMemcpyHostToDevice));

    dim3 block_dim_accum(1024);
    dim3 grid_dim_accum(n_attn_heads, (head_dim + block_dim_accum.x - 1) / block_dim_accum.x);
    hipLaunchKernelGGL(weighted_value_accumulation_kernel, grid_dim_accum, block_dim_accum, 0, 0,
                       att_scores_d, v_cache_d, tb_d, pos, seq_len, head_dim, kv_dim, kv_mul,
                       n_attn_heads);

    CHECK_HIP(hipMemcpy(tb, tb_d, n_attn_heads * head_dim * sizeof(float), hipMemcpyDeviceToHost));

    CHECK_HIP(hipFree(att_scores_d));
    CHECK_HIP(hipFree(v_cache_d));
    CHECK_HIP(hipFree(tb_d));
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
}

// ! HYRBID
void compute_attention_scores_hip_device(float* q, float* k_cache, float* att_scores, float* mask,
                                         int pos, int seq_len, int head_dim, int kv_dim, int kv_mul,
                                         int sliding_window, int layer_idx, int n_attn_heads) {
    CHECK_HIP(hipMemset(att_scores, 0, n_attn_heads * seq_len * sizeof(float)));

    dim3 block_dim_attn(256);
    dim3 grid_dim_attn(n_attn_heads, pos + 1);
    hipLaunchKernelGGL(compute_attention_scores_kernel, grid_dim_attn, block_dim_attn, 0, 0, q,
                       k_cache, att_scores, mask, pos, seq_len, head_dim, kv_dim, kv_mul,
                       sliding_window, layer_idx, n_attn_heads);

    CHECK_HIP(hipGetLastError());
}

// Kernel to convert FP16 attention sinks to FP32
__global__ void convert_attn_sinks_kernel(__half* attn_sinks_half, float* attn_sinks_fp32,
                                          int n_attn_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_attn_heads) {
        attn_sinks_fp32[idx] = __half2float(attn_sinks_half[idx]);
    }
}

void add_attention_sink_hip_device(float* att_scores, __half* attn_sinks_half, int pos, int seq_len,
                                   int n_attn_heads, int layer_idx) {
    // Convert FP16 attention sinks to FP32 on-the-fly
    if (n_attn_heads <= 0 || pos < 0 || seq_len <= 0) {
        printf("ATTN SINK DEVICE ERROR: INVALID ARGUMENT\n");
        fflush(stdout);
        return;
    }

    // Allocate temporary FP32 buffer for converted attention sinks
    float* attn_sinks_fp32;
    CHECK_HIP(hipMalloc(&attn_sinks_fp32, n_attn_heads * sizeof(float)));

    // Convert FP16 to FP32
    dim3 convert_block(64);
    dim3 convert_grid((n_attn_heads + 63) / 64);
    hipLaunchKernelGGL(convert_attn_sinks_kernel, convert_grid, convert_block, 0, 0,
                       attn_sinks_half, attn_sinks_fp32, n_attn_heads);

    dim3 block_dim_sink(1);
    dim3 grid_dim_sink(n_attn_heads);
    hipLaunchKernelGGL(add_attention_sink_kernel, grid_dim_sink, block_dim_sink, 0, 0, att_scores,
                       attn_sinks_fp32, pos, seq_len, n_attn_heads, layer_idx);

    CHECK_HIP(hipFree(attn_sinks_fp32));
    CHECK_HIP(hipGetLastError());
}

void softmax_attention_hip_device(float* att_scores, int pos, int seq_len, int n_attn_heads) {
    for (int h = 0; h < n_attn_heads; h++) {
        float* att_head_d = att_scores + h * seq_len;
        dim3 blockDim(1024);
        dim3 gridDim(1);
        hipLaunchKernelGGL(attention_softmax_kernel, gridDim, blockDim, 0, 0, att_head_d, pos + 2);
    }

    CHECK_HIP(hipGetLastError());
}

void weighted_value_accumulation_hip_device(float* att_scores, float* v_cache, float* tb, int pos,
                                            int seq_len, int head_dim, int kv_dim, int kv_mul,
                                            int n_attn_heads) {
    dim3 block_dim_accum(1024);
    dim3 grid_dim_accum(n_attn_heads, (head_dim + block_dim_accum.x - 1) / block_dim_accum.x);
    hipLaunchKernelGGL(weighted_value_accumulation_kernel, grid_dim_accum, block_dim_accum, 0, 0,
                       att_scores, v_cache, tb, pos, seq_len, head_dim, kv_dim, kv_mul,
                       n_attn_heads);

    CHECK_HIP(hipGetLastError());
}
