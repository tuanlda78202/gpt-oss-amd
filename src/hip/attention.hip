#include "BLAS.hip"
#include <cmath>

// ====== SOFTMAX KERNEL ======
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

// ! HYRBID
void compute_attn_gpu(float* q, float* k_cache, float* att_scores, float* mask,
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

void add_attn_sink_gpu(float* att_scores, __half* attn_sinks_half, int pos, int seq_len,
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

void softmax_attn_gpu(float* att_scores, int pos, int seq_len, int n_attn_heads) {
    for (int h = 0; h < n_attn_heads; h++) {
        float* att_head_d = att_scores + h * seq_len;
        dim3 blockDim(1024);
        dim3 gridDim(1);
        hipLaunchKernelGGL(attention_softmax_kernel, gridDim, blockDim, 0, 0, att_head_d, pos + 2);
    }

    CHECK_HIP(hipGetLastError());
}

void w_value_acc_gpu(float* att_scores, float* v_cache, float* tb, int pos,
                                            int seq_len, int head_dim, int kv_dim, int kv_mul,
                                            int n_attn_heads) {
    dim3 block_dim_accum(1024);
    dim3 grid_dim_accum(n_attn_heads, (head_dim + block_dim_accum.x - 1) / block_dim_accum.x);
    hipLaunchKernelGGL(weighted_value_accumulation_kernel, grid_dim_accum, block_dim_accum, 0, 0,
                       att_scores, v_cache, tb, pos, seq_len, head_dim, kv_dim, kv_mul,
                       n_attn_heads);

    CHECK_HIP(hipGetLastError());
}

// ============================================================
//                FLASH ATTENTION
//  * Numerically-stable online softmax (masked-step safe)
//  * Preload & pre-scale Q into shared memory (1× load)
//  * Variable block size (64..256) to match head_dim
//  * Optional FP16 KV-cache path (set USE_FP16_KV 1)
//  * Multi-CTA chunking across time with associative merge
// ============================================================
#ifndef USE_FP16_KV
#define USE_FP16_KV 0
#endif

__global__ void flash_attn_decode_kernel(const float* __restrict__ q,
                                         const void*  __restrict__ k_cache,
                                         const void*  __restrict__ v_cache,
                                         const float* __restrict__ mask,
                                         const __half* __restrict__ attn_sinks_half,
                                         float* __restrict__ tb,
                                         int pos, int seq_len, int head_dim, int kv_dim,
                                         int kv_mul, int sliding_window, int layer_idx,
                                         int n_attn_heads);

// Each CTA processes a time CHUNK and outputs (m_c, l_c, O_c)
__launch_bounds__(256, 2)
__global__ void flash_attn_decode_chunk_kernel(const float* __restrict__ q,
                                               const void*  __restrict__ k_cache,
                                               const void*  __restrict__ v_cache,
                                               const float* __restrict__ mask,
                                               int pos, int seq_len, int head_dim, int kv_dim,
                                               int kv_mul, int sliding_window, int layer_idx,
                                               int n_attn_heads,
                                               int chunk_size, int n_chunks,
                                               // outputs
                                               float* __restrict__ partial_O, // [H, C, D]
                                               float* __restrict__ partial_m, // [H, C]
                                               float* __restrict__ partial_l  // [H, C]
                                               ) {
    const int h  = blockIdx.x;           // head
    const int c  = blockIdx.y;           // chunk index
    const int tid = threadIdx.x;
    if (h >= n_attn_heads || c >= n_chunks) return;

    const int g = h / kv_mul; // GQA group index

    // Chunk time range
    const int t0 = c * chunk_size;
    int t1 = t0 + chunk_size; if (t1 > (pos + 1)) t1 = pos + 1;
    if (t0 >= t1) {
        if (tid == 0) { // mark empty
            partial_m[h * n_chunks + c] = -3.402823466e+38f;
            partial_l[h * n_chunks + c] = 0.0f;
        }
        return;
    }

    extern __shared__ float q_sh[]; // D floats
    const float* q_head = q + h * head_dim;

    // Preload & pre-scale Q once
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int i = tid; i < head_dim; i += blockDim.x) q_sh[i] = q_head[i] * inv_sqrt_d;
    __syncthreads();

    // Output pointer for this (h,c)
    float* O_c = partial_O + ( (h * n_chunks + c) * head_dim );

    // Initialize
    for (int i = tid; i < head_dim; i += blockDim.x) O_c[i] = 0.0f;

    __shared__ float m_shared; // running max
    __shared__ float l_shared; // running sum exp
    __shared__ float scale_old_shared, scale_new_shared, s_shared;
    __shared__ int masked_step;

    if (tid == 0) { m_shared = -3.402823466e+38f; l_shared = 0.0f; }
    __syncthreads();

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0);

    for (int t = t0; t < t1; ++t) {
#if USE_FP16_KV
        const __half* __restrict__ k_head_h = (const __half*)k_cache + t * kv_dim + g * head_dim;
#else
        const float*  __restrict__ k_head_f = (const float*) k_cache + t * kv_dim + g * head_dim;
#endif
        float partial = 0.0f;
#if USE_FP16_KV
        int i = tid * 2;
        for (; i + 1 < head_dim; i += blockDim.x * 2) {
            __half2 kh2 = *reinterpret_cast<const __half2*>(&k_head_h[i]);
            float2 kf = __half22float2(kh2);
            partial += q_sh[i] * kf.x + q_sh[i + 1] * kf.y;
        }
        for (; i < head_dim; i += blockDim.x) partial += q_sh[i] * __half2float(k_head_h[i]);
#else
        for (int i = tid; i < head_dim; i += blockDim.x) partial += q_sh[i] * k_head_f[i];
#endif
        partial = block_reduce_sum(partial);

        if (tid == 0) {
            float s = partial;
            if (use_mask) s += mask[pos * seq_len + t];
            s_shared = s; masked_step = !isfinite(s);
        }
        __syncthreads();
        if (masked_step) continue;

        if (tid == 0) {
            const float s = s_shared;
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, s);
            const float alpha  = __expf(m_prev - m_new);
            const float p      = __expf(s - m_new);
            const float l_new  = alpha * l_prev + p;
            scale_old_shared = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new_shared = (l_new > 0.0f) ? p / l_new : 0.0f;
            m_shared = m_new; l_shared = l_new;
        }
        __syncthreads();

#if USE_FP16_KV
        const __half* __restrict__ v_head_h = (const __half*)v_cache + t * kv_dim + g * head_dim;
#else
        const float*  __restrict__ v_head_f = (const float*) v_cache + t * kv_dim + g * head_dim;
#endif
        const float scale_old = scale_old_shared;
        const float scale_new = scale_new_shared;

#if USE_FP16_KV
        int j = tid * 2;
        for (; j + 1 < head_dim; j += blockDim.x * 2) {
            __half2 vh2 = *reinterpret_cast<const __half2*>(&v_head_h[j]);
            float2 vf = __half22float2(vh2);
            float o0 = O_c[j]; float o1 = O_c[j + 1];
            o0 = scale_old * o0 + scale_new * vf.x;
            o1 = scale_old * o1 + scale_new * vf.y;
            O_c[j] = o0; O_c[j + 1] = o1;
        }
        for (; j < head_dim; j += blockDim.x) {
            float o = O_c[j];
            o = scale_old * o + scale_new * __half2float(v_head_h[j]);
            O_c[j] = o;
        }
#else
        for (int j = tid; j < head_dim; j += blockDim.x) {
            float o = O_c[j];
            o = scale_old * o + scale_new * v_head_f[j];
            O_c[j] = o;
        }
#endif
    }

    if (tid == 0) {
        partial_m[h * n_chunks + c] = m_shared;
        partial_l[h * n_chunks + c] = l_shared;
    }
}

// Reduce (m,l,O) over chunks per head, then apply sink
__launch_bounds__(256, 2)
__global__ void flash_attn_reduce_chunks_kernel(float* __restrict__ partial_O,
                                                float* __restrict__ partial_m,
                                                float* __restrict__ partial_l,
                                                const __half* __restrict__ attn_sinks_half,
                                                float* __restrict__ tb,
                                                int head_dim, int n_chunks, int n_attn_heads) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    if (h >= n_attn_heads) return;

    // Initialize with chunk 0
    float m = partial_m[h * n_chunks + 0];
    float l = partial_l[h * n_chunks + 0];
    float* O_acc = tb + h * head_dim;

    // Copy O0
    for (int i = tid; i < head_dim; i += blockDim.x) {
        O_acc[i] = partial_O[(h * n_chunks + 0) * head_dim + i];
    }
    __syncthreads();

    // Merge remaining chunks
    for (int c = 1; c < n_chunks; ++c) {
        float mc = partial_m[h * n_chunks + c];
        float lc = partial_l[h * n_chunks + c];
        const float* Oc = partial_O + ( (h * n_chunks + c) * head_dim );
        // Skip empty chunk
        if (!(isfinite(mc) && lc > 0.0f)) continue;

        float m_new = fmaxf(m, mc);
        float a = __expf(m - m_new) * l;   // weight A
        float b = __expf(mc - m_new) * lc; // weight B
        float denom = a + b;
        float wA = (denom > 0.0f) ? a / denom : 0.0f;
        float wB = (denom > 0.0f) ? b / denom : 0.0f;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            float o = wA * O_acc[i] + wB * Oc[i];
            O_acc[i] = o;
        }
        __syncthreads();
        m = m_new; l = denom;
    }

    // Apply attention sink (denominator only)
    if (attn_sinks_half != nullptr) {
        float sink_s = __half2float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            float m_new = fmaxf(m, sink_s);
            float a = __expf(m - m_new) * l;     // carries O
            float b = __expf(sink_s - m_new);    // V_sink = 0
            float denom = a + b;
            float renorm = (denom > 0.0f) ? (a / denom) : 1.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) {
                O_acc[i] *= renorm;
            }
        }
    }
}

// Single-CTA-per-head variant kept for short contexts
__launch_bounds__(256, 2)
__global__ void flash_attn_decode_kernel(const float* __restrict__ q,
                                         const void*  __restrict__ k_cache,
                                         const void*  __restrict__ v_cache,
                                         const float* __restrict__ mask,
                                         const __half* __restrict__ attn_sinks_half,
                                         float* __restrict__ tb,
                                         int pos, int seq_len, int head_dim, int kv_dim,
                                         int kv_mul, int sliding_window, int layer_idx,
                                         int n_attn_heads) {
    const int h = blockIdx.x;            // head
    const int tid = threadIdx.x;
    if (h >= n_attn_heads) return;

    extern __shared__ float q_sh[];
    const int g = h / kv_mul;
    const float* q_head = q + h * head_dim;
    float* tb_head = tb + h * head_dim;

    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int i = tid; i < head_dim; i += blockDim.x) q_sh[i] = q_head[i] * inv_sqrt_d;
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) tb_head[i] = 0.0f;

    __shared__ float m_shared, l_shared, scale_old_shared, scale_new_shared, s_shared;
    __shared__ int masked_step;
    if (tid == 0) { m_shared = -3.402823466e+38f; l_shared = 0.0f; }
    __syncthreads();

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0);

    for (int t = 0; t <= pos; ++t) {
#if USE_FP16_KV
        const __half* __restrict__ k_head_h = (const __half*)k_cache + t * kv_dim + g * head_dim;
#else
        const float*  __restrict__ k_head_f = (const float*) k_cache + t * kv_dim + g * head_dim;
#endif
        float partial = 0.0f;
#if USE_FP16_KV
        int i = tid * 2;
        for (; i + 1 < head_dim; i += blockDim.x * 2) {
            __half2 kh2 = *reinterpret_cast<const __half2*>(&k_head_h[i]);
            float2 kf = __half22float2(kh2);
            partial += q_sh[i] * kf.x + q_sh[i + 1] * kf.y;
        }
        for (; i < head_dim; i += blockDim.x) partial += q_sh[i] * __half2float(k_head_h[i]);
#else
        for (int i = tid; i < head_dim; i += blockDim.x) partial += q_sh[i] * k_head_f[i];
#endif
        partial = block_reduce_sum(partial);

        if (tid == 0) {
            float s = partial; if (use_mask) s += mask[pos * seq_len + t];
            s_shared = s; masked_step = !isfinite(s);
        }
        __syncthreads();
        if (masked_step) continue;

        if (tid == 0) {
            const float s = s_shared;
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, s);
            const float alpha  = __expf(m_prev - m_new);
            const float p      = __expf(s - m_new);
            const float l_new  = alpha * l_prev + p;
            scale_old_shared = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new_shared = (l_new > 0.0f) ? p / l_new : 0.0f;
            m_shared = m_new; l_shared = l_new;
        }
        __syncthreads();

#if USE_FP16_KV
        const __half* __restrict__ v_head_h = (const __half*)v_cache + t * kv_dim + g * head_dim;
#else
        const float*  __restrict__ v_head_f = (const float*) v_cache + t * kv_dim + g * head_dim;
#endif
        const float scale_old = scale_old_shared;
        const float scale_new = scale_new_shared;

#if USE_FP16_KV
        int j = tid * 2;
        for (; j + 1 < head_dim; j += blockDim.x * 2) {
            __half2 vh2 = *reinterpret_cast<const __half2*>(&v_head_h[j]);
            float2 vf = __half22float2(vh2);
            float o0 = tb_head[j]; float o1 = tb_head[j + 1];
            o0 = scale_old * o0 + scale_new * vf.x;
            o1 = scale_old * o1 + scale_new * vf.y;
            tb_head[j] = o0; tb_head[j + 1] = o1;
        }
        for (; j < head_dim; j += blockDim.x) {
            float o = tb_head[j];
            o = scale_old * o + scale_new * __half2float(v_head_h[j]);
            tb_head[j] = o;
        }
#else
        for (int j = tid; j < head_dim; j += blockDim.x) {
            float o = tb_head[j];
            o = scale_old * o + scale_new * v_head_f[j];
            tb_head[j] = o;
        }
#endif
    }

    // sink
    if (attn_sinks_half != nullptr) {
        float sink_s = __half2float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m_shared; const float l_prev = l_shared;
            float m_new = fmaxf(m_prev, sink_s);
            float a = __expf(m_prev - m_new) * l_prev;
            float b = __expf(sink_s - m_new);
            float denom = a + b; float renorm = (denom > 0.0f) ? (a / denom) : 1.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) tb_head[i] *= renorm;
        }
    }
}

static inline int fa_next_pow2(int x){ int p=1; while(p<x) p<<=1; return p; }

// Heuristic: chunk if we can spawn >= 4*#heads CTAs
static inline int choose_chunk_size(int L, int heads, int target_cta_per_head=8) {
    if (L <= 256) return L; // single CTA likely fine
    int c = 256; // good default on gfx90a
    int chunks = (L + c - 1) / c;
    if (chunks < target_cta_per_head) c = 128; // more CTAs
    return c;
}

void flash_attn_decode_gpu(const float* q,
                           const float* k_cache, const float* v_cache,
                           const float* mask, const __half* attn_sinks_half,
                           float* tb, int pos, int seq_len, int head_dim, int kv_dim,
                           int kv_mul, int sliding_window, int layer_idx, int n_attn_heads) {
    if (n_attn_heads <= 0 || head_dim <= 0) return;

    const int L = pos + 1; // tokens to attend
    int chunk_size = choose_chunk_size(L, n_attn_heads);
    int n_chunks = (L + chunk_size - 1) / chunk_size;

    // Pick threads per block (64..256) matching head_dim
    int tpb = fa_next_pow2(head_dim); if (tpb < 64) tpb = 64; else if (tpb > 256) tpb = 256;
    size_t shmem = (size_t)head_dim * sizeof(float);

    if (n_chunks <= 1) {
        dim3 grid(n_attn_heads), block(tpb);
        hipLaunchKernelGGL(flash_attn_decode_kernel, grid, block, shmem, 0,
                           q, (const void*)k_cache, (const void*)v_cache, mask, attn_sinks_half,
                           tb, pos, seq_len, head_dim, kv_dim, kv_mul,
                           sliding_window, layer_idx, n_attn_heads);
        CHECK_HIP(hipGetLastError());
        return;
    }

    // Allocate partials: [H, C, D] and [H, C]
    float* partial_O; float* partial_m; float* partial_l;
    CHECK_HIP(hipMalloc(&partial_O, (size_t)n_attn_heads * n_chunks * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&partial_m, (size_t)n_attn_heads * n_chunks * sizeof(float)));
    CHECK_HIP(hipMalloc(&partial_l, (size_t)n_attn_heads * n_chunks * sizeof(float)));

    // Kernel 1: process chunks
    dim3 grid1(n_attn_heads, n_chunks), block1(tpb);
    hipLaunchKernelGGL(flash_attn_decode_chunk_kernel, grid1, block1, shmem, 0,
                       q, (const void*)k_cache, (const void*)v_cache, mask,
                       pos, seq_len, head_dim, kv_dim, kv_mul,
                       sliding_window, layer_idx, n_attn_heads,
                       chunk_size, n_chunks,
                       partial_O, partial_m, partial_l);

    // Kernel 2: reduce across chunks + apply sink
    dim3 grid2(n_attn_heads), block2(tpb);
    hipLaunchKernelGGL(flash_attn_reduce_chunks_kernel, grid2, block2, 0, 0,
                       partial_O, partial_m, partial_l, attn_sinks_half,
                       tb, head_dim, n_chunks, n_attn_heads);

    CHECK_HIP(hipFree(partial_O));
    CHECK_HIP(hipFree(partial_m));
    CHECK_HIP(hipFree(partial_l));
    CHECK_HIP(hipGetLastError());
}

// ============================================================
// Batched CHUNK kernel: grid = (n_attn_heads, n_chunks, batch_size)
// Each CTA handles one (head, chunk, batch) triple and writes
// partial (O_c, m_c, l_c) at indices [b, h, c, ...]
// ============================================================
__launch_bounds__(256, 2)
__global__ void flash_attn_decode_chunk_kernel_batched(
    const float* __restrict__ q_batch,      // (B, H*D)
    const void*  __restrict__ k_cache_base, // (L, B, T, kv_dim) laid out per your strides
    const void*  __restrict__ v_cache_base, // same layout as K
    const float* __restrict__ mask,         // (T, T), shared across batch
    int pos, int seq_len, int head_dim, int kv_dim,
    int kv_mul, int sliding_window, int layer_idx,
    int n_attn_heads,
    int chunk_size, int n_chunks,
    // batch routing/strides
    const int* __restrict__ batch_indices,  // (B) -> global-batch id
    long long B_stride,                     // elements per (B * T * kv_dim) “layer stride”
    // outputs
    float* __restrict__ partial_O, // (B, H, C, D)
    float* __restrict__ partial_m, // (B, H, C)
    float* __restrict__ partial_l  // (B, H, C)
) {
    const int h   = blockIdx.x;  // head
    const int c   = blockIdx.y;  // chunk
    const int b   = blockIdx.z;  // batch
    const int tid = threadIdx.x;

    if (h >= n_attn_heads || c >= n_chunks) return;

    const int g = h / kv_mul; // GQA group index

    // Chunk time range
    const int t0 = c * chunk_size;
    int t1 = t0 + chunk_size; if (t1 > (pos + 1)) t1 = pos + 1;
    if (t0 >= t1) {
        if (tid == 0) {
            partial_m[((long long)b*n_attn_heads + h)*n_chunks + c] = -3.402823466e+38f;
            partial_l[((long long)b*n_attn_heads + h)*n_chunks + c] = 0.0f;
        }
        return;
    }

    // Per-batch pointers for Q and partial outputs
    const float* q_b = q_batch + (long long)b * (n_attn_heads * head_dim);
    float* O_c = partial_O + ( ((long long)b*n_attn_heads + h) * n_chunks + c ) * (long long)head_dim;

    extern __shared__ float q_sh[]; // D floats
    const float* q_head = q_b + (long long)h * head_dim;

    // Preload & pre-scale Q once
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int i = tid; i < head_dim; i += blockDim.x) q_sh[i] = q_head[i] * inv_sqrt_d;
    __syncthreads();

    // Initialize O_c
    for (int i = tid; i < head_dim; i += blockDim.x) O_c[i] = 0.0f;

    __shared__ float m_shared; // running max
    __shared__ float l_shared; // running sum exp
    __shared__ float scale_old_shared, scale_new_shared, s_shared;
    __shared__ int masked_step;

    if (tid == 0) { m_shared = -3.402823466e+38f; l_shared = 0.0f; }
    __syncthreads();

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0);

    // Compute KV base for this (layer, gb)
    const int gb = batch_indices[b];
    const long long layer_stride = 1ll * B_stride * seq_len * kv_dim;  // ✅ elements
    const long long batch_stride = 1ll * seq_len * kv_dim;             // ✅ elements

    const size_t elem_bytes =
    #if USE_FP16_KV
        sizeof(__half);
    #else
        sizeof(float);
    #endif

    const char* k_base = (const char*)k_cache_base + elem_bytes * (
        (long long)layer_idx * layer_stride + (long long)gb * batch_stride );
    const char* v_base = (const char*)v_cache_base + elem_bytes * (
        (long long)layer_idx * layer_stride + (long long)gb * batch_stride );

    for (int t = t0; t < t1; ++t) {
#if USE_FP16_KV
        const __half* __restrict__ k_head_h = (const __half*)( (const __half*)k_base + (long long)t * kv_dim + (long long)g * head_dim );
#else
        const float*  __restrict__ k_head_f = (const float*)( (const float*)k_base + (long long)t * kv_dim + (long long)g * head_dim );
#endif
        float partial = 0.0f;
#if USE_FP16_KV
        int i = tid * 2;
        for (; i + 1 < head_dim; i += blockDim.x * 2) {
            __half2 kh2 = *reinterpret_cast<const __half2*>(&k_head_h[i]);
            float2 kf = __half22float2(kh2);
            partial += q_sh[i] * kf.x + q_sh[i + 1] * kf.y;
        }
        for (; i < head_dim; i += blockDim.x) partial += q_sh[i] * __half2float(k_head_h[i]);
#else
        for (int i = tid; i < head_dim; i += blockDim.x) partial += q_sh[i] * k_head_f[i];
#endif
        partial = block_reduce_sum(partial);

        if (tid == 0) {
            float s = partial;
            if (use_mask) s += mask[(long long)pos * seq_len + t];
            s_shared = s; masked_step = !isfinite(s);
        }
        __syncthreads();
        if (masked_step) continue;

        if (tid == 0) {
            const float s = s_shared;
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, s);
            const float alpha  = __expf(m_prev - m_new);
            const float p      = __expf(s - m_new);
            const float l_new  = alpha * l_prev + p;
            scale_old_shared = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new_shared = (l_new > 0.0f) ? p / l_new : 0.0f;
            m_shared = m_new; l_shared = l_new;
        }
        __syncthreads();

#if USE_FP16_KV
        const __half* __restrict__ v_head_h = (const __half*)( (const __half*)v_base + (long long)t * kv_dim + (long long)g * head_dim );
#else
        const float*  __restrict__ v_head_f = (const float*)( (const float*)v_base + (long long)t * kv_dim + (long long)g * head_dim );
#endif
        const float scale_old = scale_old_shared;
        const float scale_new = scale_new_shared;

#if USE_FP16_KV
        int j = tid * 2;
        for (; j + 1 < head_dim; j += blockDim.x * 2) {
            __half2 vh2 = *reinterpret_cast<const __half2*>(&v_head_h[j]);
            float2 vf = __half22float2(vh2);
            float o0 = O_c[j]; float o1 = O_c[j + 1];
            o0 = scale_old * o0 + scale_new * vf.x;
            o1 = scale_old * o1 + scale_new * vf.y;
            O_c[j] = o0; O_c[j + 1] = o1;
        }
        for (; j < head_dim; j += blockDim.x) {
            float o = O_c[j];
            o = scale_old * o + scale_new * __half2float(v_head_h[j]);
            O_c[j] = o;
        }
#else
        for (int j = tid; j < head_dim; j += blockDim.x) {
            float o = O_c[j];
            o = scale_old * o + scale_new * v_head_f[j];
            O_c[j] = o;
        }
#endif
    }

    if (tid == 0) {
        partial_m[((long long)b*n_attn_heads + h)*n_chunks + c] = m_shared;
        partial_l[((long long)b*n_attn_heads + h)*n_chunks + c] = l_shared;
    }
}

// ============================================================
// Batched REDUCE kernel: grid = (n_attn_heads, batch_size)
// Reduces over chunks for each (b, h), writes to tb_batch[b, h, :]
// Also applies attention sink per (b, h)
// ============================================================
__launch_bounds__(256, 2)
__global__ void flash_attn_reduce_chunks_kernel_batched(
    const float* __restrict__ partial_O,     // (B, H, C, D)
    const float* __restrict__ partial_m,     // (B, H, C)
    const float* __restrict__ partial_l,     // (B, H, C)
    const __half* __restrict__ attn_sinks_half, // (L, H) — we index by head only here
    float* __restrict__ tb_batch,            // (B, H*D)
    int head_dim, int n_chunks, int n_attn_heads
) {
    const int h   = blockIdx.x;
    const int b   = blockIdx.y;
    const int tid = threadIdx.x;
    if (h >= n_attn_heads) return;

    float m = partial_m[((long long)b*n_attn_heads + h)*n_chunks + 0];
    float l = partial_l[((long long)b*n_attn_heads + h)*n_chunks + 0];

    float* O_acc = tb_batch + (long long)b * (n_attn_heads * head_dim) + (long long)h * head_dim;

    // Copy O0
    const float* O0 = partial_O + ( ((long long)b*n_attn_heads + h) * n_chunks + 0 ) * (long long)head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) O_acc[i] = O0[i];
    __syncthreads();

    // Merge remaining chunks
    for (int c = 1; c < n_chunks; ++c) {
        const float mc = partial_m[((long long)b*n_attn_heads + h)*n_chunks + c];
        const float lc = partial_l[((long long)b*n_attn_heads + h)*n_chunks + c];
        if (!(isfinite(mc) && lc > 0.0f)) continue;

        const float* Oc = partial_O + ( ((long long)b*n_attn_heads + h) * n_chunks + c ) * (long long)head_dim;

        const float m_new = fmaxf(m, mc);
        const float a = __expf(m  - m_new) * l;
        const float b2= __expf(mc - m_new) * lc;
        const float denom = a + b2;
        const float wA = (denom > 0.0f) ? a  / denom : 0.0f;
        const float wB = (denom > 0.0f) ? b2 / denom : 0.0f;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            float o = wA * O_acc[i] + wB * Oc[i];
            O_acc[i] = o;
        }
        __syncthreads();
        m = m_new; l = denom;
    }

    // Apply attention sink (denominator only). attn_sinks_half is per-head.
    if (attn_sinks_half != nullptr) {
        float sink_s = __half2float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m; const float l_prev = l;
            const float m_new  = fmaxf(m_prev, sink_s);
            const float a = __expf(m_prev - m_new) * l_prev;
            const float b3= __expf(sink_s - m_new);
            const float denom = a + b3;
            const float renorm = (denom > 0.0f) ? (a / denom) : 1.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) O_acc[i] *= renorm;
        }
    }
}
__launch_bounds__(256, 2)
__global__ void flash_attn_decode_kernel_batched(
    const float* __restrict__ q_batch,      // (B, H*D)
    const void*  __restrict__ k_cache_base, // (L, B, T, kv_dim)
    const void*  __restrict__ v_cache_base, // (L, B, T, kv_dim)
    const float* __restrict__ mask,         // (T, T)
    const __half* __restrict__ attn_sinks_half, // (H)
    float* __restrict__ tb_batch,           // (B, H*D)
    int pos, int seq_len, int head_dim, int kv_dim,
    int kv_mul, int sliding_window, int layer_idx, int n_attn_heads,
    const int* __restrict__ batch_indices, long long B_stride
) {
    const int h   = blockIdx.x;
    const int b   = blockIdx.y;
    const int tid = threadIdx.x;
    if (h >= n_attn_heads) return;

    const int g = h / kv_mul;

    const float* q_b  = q_batch   + (long long)b * (n_attn_heads * head_dim);
    float*       tb_b = tb_batch  + (long long)b * (n_attn_heads * head_dim);
    const float* q_head = q_b + (long long)h * head_dim;
    float* tb_head = tb_b + (long long)h * head_dim;

    extern __shared__ float q_sh[];
    const float inv_sqrt_d = rsqrtf((float)head_dim);
    for (int i = tid; i < head_dim; i += blockDim.x) q_sh[i] = q_head[i] * inv_sqrt_d;
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) tb_head[i] = 0.0f;

    __shared__ float m_shared, l_shared, scale_old_shared, scale_new_shared, s_shared;
    __shared__ int masked_step;
    if (tid == 0) { m_shared = -3.402823466e+38f; l_shared = 0.0f; }
    __syncthreads();

    const bool use_mask = (sliding_window > 0) && ((layer_idx & 1) == 0);

    const int gb = batch_indices[b];
    const long long layer_stride = 1ll * B_stride * seq_len * kv_dim;  // ✅ elements
    const long long batch_stride = 1ll * seq_len * kv_dim;             // ✅ elements

    const size_t elem_bytes =
    #if USE_FP16_KV
        sizeof(__half);
    #else
        sizeof(float);
    #endif

    const char* k_base = (const char*)k_cache_base + elem_bytes * (
        (long long)layer_idx * layer_stride + (long long)gb * batch_stride );
    const char* v_base = (const char*)v_cache_base + elem_bytes * (
        (long long)layer_idx * layer_stride + (long long)gb * batch_stride );


    for (int t = 0; t <= pos; ++t) {
#if USE_FP16_KV
        const __half* __restrict__ k_head_h = (const __half*)( (const __half*)k_base + (long long)t * kv_dim + (long long)g * head_dim );
#else
        const float*  __restrict__ k_head_f = (const float*)( (const float*)k_base + (long long)t * kv_dim + (long long)g * head_dim );
#endif
        float partial = 0.0f;
#if USE_FP16_KV
        int i = tid * 2;
        for (; i + 1 < head_dim; i += blockDim.x * 2) {
            __half2 kh2 = *reinterpret_cast<const __half2*>(&k_head_h[i]);
            float2 kf = __half22float2(kh2);
            partial += q_sh[i] * kf.x + q_sh[i + 1] * kf.y;
        }
        for (; i < head_dim; i += blockDim.x) partial += q_sh[i] * __half2float(k_head_h[i]);
#else
        for (int i = tid; i < head_dim; i += blockDim.x) partial += q_sh[i] * k_head_f[i];
#endif
        partial = block_reduce_sum(partial);

        if (tid == 0) {
            float s = partial; if (use_mask) s += mask[(long long)pos * seq_len + t];
            s_shared = s; masked_step = !isfinite(s);
        }
        __syncthreads();
        if (masked_step) continue;

        if (tid == 0) {
            const float s = s_shared;
            const float m_prev = m_shared;
            const float l_prev = l_shared;
            const float m_new  = fmaxf(m_prev, s);
            const float alpha  = __expf(m_prev - m_new);
            const float p      = __expf(s - m_new);
            const float l_new  = alpha * l_prev + p;
            scale_old_shared = (l_new > 0.0f) ? (alpha * l_prev) / l_new : 0.0f;
            scale_new_shared = (l_new > 0.0f) ? p / l_new : 0.0f;
            m_shared = m_new; l_shared = l_new;
        }
        __syncthreads();

#if USE_FP16_KV
        const __half* __restrict__ v_head_h = (const __half*)( (const __half*)v_base + (long long)t * kv_dim + (long long)g * head_dim );
#else
        const float*  __restrict__ v_head_f = (const float*)( (const float*)v_base + (long long)t * kv_dim + (long long)g * head_dim );
#endif
        const float scale_old = scale_old_shared;
        const float scale_new = scale_new_shared;

#if USE_FP16_KV
        int j = tid * 2;
        for (; j + 1 < head_dim; j += blockDim.x * 2) {
            __half2 vh2 = *reinterpret_cast<const __half2*>(&v_head_h[j]);
            float2 vf = __half22float2(vh2);
            float o0 = tb_head[j]; float o1 = tb_head[j + 1];
            o0 = scale_old * o0 + scale_new * vf.x;
            o1 = scale_old * o1 + scale_new * vf.y;
            tb_head[j] = o0; tb_head[j + 1] = o1;
        }
        for (; j < head_dim; j += blockDim.x) {
            float o = tb_head[j];
            o = scale_old * o + scale_new * __half2float(v_head_h[j]);
            tb_head[j] = o;
        }
#else
        for (int j = tid; j < head_dim; j += blockDim.x) {
            float o = tb_head[j];
            o = scale_old * o + scale_new * v_head_f[j];
            tb_head[j] = o;
        }
#endif
    }

    // sink
    if (attn_sinks_half != nullptr) {
        float sink_s = __half2float(attn_sinks_half[h]);
        if (isfinite(sink_s)) {
            const float m_prev = m_shared; const float l_prev = l_shared;
            float m_new = fmaxf(m_prev, sink_s);
            float a = __expf(m_prev - m_new) * l_prev;
            float b3= __expf(sink_s - m_new);
            float denom = a + b3; float renorm = (denom > 0.0f) ? (a / denom) : 1.0f;
            for (int i = tid; i < head_dim; i += blockDim.x) tb_head[i] *= renorm;
        }
    }
}

void flash_attn_decode_gpu_batch(
    const float* q_batch,      // (B, H*D)
    const float* k_cache,      // base pointer (L, B, T, kv_dim) by your stride math
    const float* v_cache,
    const float* mask,         // (T, T)
    const __half* attn_sinks,  // (L, H) — we use attn_sinks + layer_idx*H
    float* tb_batch,           // (B, H*D)
    int batch_size,
    int pos, int seq_len, int head_dim, int kv_dim, int kv_mul,
    int sliding_window, int layer_idx, int n_attn_heads,
    const int* batch_indices,  // (B) maps logical b -> global batch slot in cache
    int B_stride               // elements in (B * T * kv_dim) per layer
){
    if (batch_size <= 0 || n_attn_heads <= 0 || head_dim <= 0) return;

    const int L = pos + 1;
    int chunk_size = choose_chunk_size(L, n_attn_heads);
    int n_chunks   = (L + chunk_size - 1) / chunk_size;

    // threads per block (64..256) matching head_dim
    int tpb = fa_next_pow2(head_dim);
    if (tpb < 64) tpb = 64; else if (tpb > 256) tpb = 256;
    size_t shmem = (size_t)head_dim * sizeof(float);

    const __half* attn_sinks_head = (attn_sinks != nullptr) ? (attn_sinks + (long long)layer_idx * n_attn_heads) : nullptr;

    if (n_chunks <= 1) {
        // Single-CTA per (head, batch)
        dim3 grid(n_attn_heads, batch_size), block(tpb);
        hipLaunchKernelGGL(flash_attn_decode_kernel_batched, grid, block, shmem, 0,
                           q_batch,
                           (const void*)k_cache, (const void*)v_cache,
                           mask, attn_sinks_head, tb_batch,
                           pos, seq_len, head_dim, kv_dim, kv_mul,
                           sliding_window, layer_idx, n_attn_heads,
                           batch_indices, (long long)B_stride);
        CHECK_HIP(hipGetLastError());
        return;
    }

    // Allocate partials for (B, H, C, D) and (B, H, C)
    float* partial_O; float* partial_m; float* partial_l;
    CHECK_HIP(hipMalloc(&partial_O, (long long)batch_size * n_attn_heads * n_chunks * head_dim * sizeof(float)));
    CHECK_HIP(hipMalloc(&partial_m, (long long)batch_size * n_attn_heads * n_chunks * sizeof(float)));
    CHECK_HIP(hipMalloc(&partial_l, (long long)batch_size * n_attn_heads * n_chunks * sizeof(float)));

    // Kernel 1: chunks (heads × chunks × batch)
    {
        dim3 grid1(n_attn_heads, n_chunks, batch_size), block1(tpb);
        hipLaunchKernelGGL(flash_attn_decode_chunk_kernel_batched, grid1, block1, shmem, 0,
                           q_batch,
                           (const void*)k_cache, (const void*)v_cache,
                           mask,
                           pos, seq_len, head_dim, kv_dim, kv_mul,
                           sliding_window, layer_idx, n_attn_heads,
                           chunk_size, n_chunks,
                           batch_indices, (long long)B_stride,
                           partial_O, partial_m, partial_l);
    }

    // Kernel 2: reduce (heads × batch)
    {
        dim3 grid2(n_attn_heads, batch_size), block2(tpb);
        hipLaunchKernelGGL(flash_attn_reduce_chunks_kernel_batched, grid2, block2, 0, 0,
                           partial_O, partial_m, partial_l,
                           attn_sinks_head, tb_batch,
                           head_dim, n_chunks, n_attn_heads);
    }

    CHECK_HIP(hipFree(partial_O));
    CHECK_HIP(hipFree(partial_m));
    CHECK_HIP(hipFree(partial_l));
    CHECK_HIP(hipGetLastError());
}
