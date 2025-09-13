#include "../include/model.hpp"

#include "../include/pp_manager.hpp"

//---------------- helpers ----------------//
static std::vector<std::pair<int, int>> partition_layers(int n_layers, int parts) {
    std::vector<std::pair<int, int>> cuts;
    int base = n_layers / parts, rem = n_layers % parts, start = 0;
    for (int p = 0; p < parts; ++p) {
        int take = base + (p < rem ? 1 : 0);
        cuts.push_back({start, start + take});
        start += take;
    }
    return cuts;
}

static bool enable_p2p(int src_dev, int dst_dev) {
    int can = 0;
    CHECK_HIP(hipDeviceCanAccessPeer(&can, dst_dev, src_dev));
    if (can) {
        CHECK_HIP(hipSetDevice(dst_dev));
        hipError_t e = hipDeviceEnablePeerAccess(src_dev, 0);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
            CHECK_HIP(e);
        }
    }
    return can != 0;
}

void copy_large_tensor_streaming(__half** d_ptr, float* h_ptr, size_t total_size,
                                 const char* tensor_name) {
    const size_t chunk_size = 512 * 1024 * 1024;

    bool show_progress = total_size >= 1024 * 1024 * 1024;

    if (show_progress) {
        printf("  %s (%.1f GB)...", tensor_name, total_size / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
    }

    // Allocate temporary conversion buffer
    __half* conversion_buffer = (__half*)malloc(chunk_size);
    if (!conversion_buffer) {
        fprintf(stderr, "Failed to allocate conversion buffer for %s\n", tensor_name);
        exit(EXIT_FAILURE);
    }

    size_t bytes_processed = 0;
    size_t chunks_processed = 0;
    size_t total_elements = total_size / sizeof(__half);

    while (bytes_processed < total_size) {
        size_t current_chunk_bytes = (total_size - bytes_processed < chunk_size)
                                         ? (total_size - bytes_processed)
                                         : chunk_size;
        size_t current_chunk_elements = current_chunk_bytes / sizeof(__half);

        size_t element_offset = bytes_processed / sizeof(__half);
        for (size_t i = 0; i < current_chunk_elements; i++) {
            conversion_buffer[i] = __float2half(h_ptr[element_offset + i]);
        }

        // Transfer converted chunk directly to GPU
        CHECK_HIP(hipMemcpy((char*)(*d_ptr) + bytes_processed, conversion_buffer,
                            current_chunk_bytes, hipMemcpyHostToDevice));

        bytes_processed += current_chunk_bytes;
        chunks_processed++;

        if (show_progress && chunks_processed % 8 == 0) {
            printf(".");
            fflush(stdout);
        }
    }

    free(conversion_buffer);

    if (show_progress) {
        printf(" done\n");
    }
}

static void copy_transformer_to_device_hybrid_pp(OssTransformer* t_fp32, OssTransformerHybrid* t_d,
                                                 int l_start, int l_end, bool has_embed,
                                                 bool has_out) {
    // Clone config but localize n_layers to just our shard
    std::memcpy(&t_d->config, &t_fp32->config, sizeof(OssConfig));
    OssConfig* conf = &t_d->config;
    conf->n_layers = l_end - l_start;

    // Short names
    OssTransformerWeights* W = &t_fp32->weights;
    int vocab = conf->vocab_size, H = conf->hidden_dim, L_local = conf->n_layers;
    int hd = conf->head_dim, nH = conf->n_attn_heads, nKV = conf->n_kv_heads;
    int kv_dim = hd * nKV;
    int inter = conf->intermediate_dim, nExp = conf->n_experts;
    int experts_per_token = conf->experts_per_token;
    int seq_len = conf->seq_len;

    int current_device;
    CHECK_HIP(hipGetDevice(&current_device));

    hipDeviceProp_t deviceProp;
    CHECK_HIP(hipGetDeviceProperties(&deviceProp, current_device));

    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    printf("GPU %d (%s): %.1f GB free / %.1f GB total\n", current_device, deviceProp.name,
           free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));

    // Allocate device weights (FP16) — only for our local layers
    auto Malloc = [](void** p, size_t n) { CHECK_HIP(hipMalloc(p, n)); };

    if (has_embed)
        Malloc((void**)&t_d->weights.token_embedding_table, (size_t)vocab * H * sizeof(__half));
    else
        t_d->weights.token_embedding_table = nullptr;

    if (has_out) {
        Malloc((void**)&t_d->weights.rms_out_w, (size_t)H * sizeof(__half));
        Malloc((void**)&t_d->weights.out, (size_t)vocab * H * sizeof(__half));
    } else {
        t_d->weights.rms_out_w = nullptr;
        t_d->weights.out = nullptr;
    }

    // Per-layer allocations (local count)
    Malloc((void**)&t_d->weights.rms_attn_w, (size_t)L_local * H * sizeof(__half));
    Malloc((void**)&t_d->weights.rms_ffn_w, (size_t)L_local * H * sizeof(__half));

    size_t qkv_perL_elems = (size_t)H * (hd * nH + 2 * hd * nKV);
    size_t wo_perL_elems = (size_t)H * (hd * nH);
    size_t bqkv_perL_elems = (size_t)(hd * nH + 2 * hd * nKV);
    size_t bo_perL_elems = (size_t)H;
    size_t sinks_perL = (size_t)nH;

    Malloc((void**)&t_d->weights.w_qkv, (size_t)L_local * qkv_perL_elems * sizeof(__half));
    Malloc((void**)&t_d->weights.w_o, (size_t)L_local * wo_perL_elems * sizeof(__half));
    Malloc((void**)&t_d->weights.b_qkv, (size_t)L_local * bqkv_perL_elems * sizeof(__half));
    Malloc((void**)&t_d->weights.b_o, (size_t)L_local * bo_perL_elems * sizeof(__half));
    Malloc((void**)&t_d->weights.attn_sinks, (size_t)L_local * sinks_perL * sizeof(__half));

    Malloc((void**)&t_d->weights.w_router, (size_t)L_local * (size_t)H * nExp * sizeof(__half));
    Malloc((void**)&t_d->weights.b_router, (size_t)L_local * (size_t)nExp * sizeof(__half));
    Malloc((void**)&t_d->weights.w_mlp1,
           (size_t)L_local * (size_t)nExp * (2 * inter) * H * sizeof(__half));
    Malloc((void**)&t_d->weights.w_mlp2,
           (size_t)L_local * (size_t)nExp * inter * H * sizeof(__half));
    Malloc((void**)&t_d->weights.b_mlp1,
           (size_t)L_local * (size_t)nExp * (2 * inter) * sizeof(__half));
    Malloc((void**)&t_d->weights.b_mlp2, (size_t)L_local * (size_t)nExp * H * sizeof(__half));

    // ! Allocate FP32 state on GPU (with batch dimension B)
    int batch_size = conf->batch_size;
    printf("🚒 Allocating GPU buffers for batch size %d...\n", batch_size);

    auto Z = [](void* p, size_t n) { CHECK_HIP(hipMemset(p, 0, n)); };
    Malloc((void**)&t_d->state.x, (size_t)batch_size * H * sizeof(float));
    Malloc((void**)&t_d->state.t, (size_t)batch_size * H * sizeof(float));
    Malloc((void**)&t_d->state.tb, (size_t)batch_size * hd * nH * sizeof(float));
    Malloc((void**)&t_d->state.tb2, (size_t)batch_size * H * sizeof(float));
    Malloc((void**)&t_d->state.router_score, (size_t)batch_size * nExp * sizeof(float));
    Malloc((void**)&t_d->state.topk_v, (size_t)batch_size * experts_per_token * sizeof(float));
    Malloc((void**)&t_d->state.topk_i, (size_t)batch_size * experts_per_token * sizeof(int));
    Malloc((void**)&t_d->state.mlp1_out, (size_t)batch_size * 2 * inter * sizeof(float));
    Malloc((void**)&t_d->state.gate, (size_t)batch_size * inter * sizeof(float));
    Malloc((void**)&t_d->state.up, (size_t)batch_size * inter * sizeof(float));
    Malloc((void**)&t_d->state.gate_up, (size_t)batch_size * inter * sizeof(float));
    Malloc((void**)&t_d->state.e_agg, (size_t)batch_size * H * sizeof(float));
    Malloc((void**)&t_d->state.qkv, (size_t)batch_size * hd * (nH + 2 * nKV) * sizeof(float));
    Malloc((void**)&t_d->state.q, (size_t)batch_size * nH * hd * sizeof(float));
    Malloc((void**)&t_d->state.att, (size_t)batch_size * nH * seq_len * sizeof(float));
    if (has_out)
        Malloc((void**)&t_d->state.logits, (size_t)batch_size * vocab * sizeof(float));
    else
        t_d->state.logits = nullptr;

    size_t key_cache_sz = (size_t)L_local * batch_size * seq_len * kv_dim * sizeof(float);
    Malloc((void**)&t_d->state.key_cache, key_cache_sz);
    Malloc((void**)&t_d->state.value_cache, key_cache_sz);
    Malloc((void**)&t_d->state.mask, (size_t)seq_len * seq_len * sizeof(float));

    Malloc((void**)&t_d->state.d_batch_indices, (size_t)batch_size * sizeof(int));
    Malloc((void**)&t_d->state.d_tokens, (size_t)batch_size * sizeof(int));
    Malloc((void**)&t_d->state.d_pos_per_token, (size_t)batch_size * sizeof(int));
    Malloc((void**)&t_d->state.cos_vals, (size_t)(hd / 2) * sizeof(float));
    Malloc((void**)&t_d->state.sin_vals, (size_t)(hd / 2) * sizeof(float));

    // MoE expert-batching scratch buffers
    int BK_max = batch_size * experts_per_token;
    Malloc((void**)&t_d->state.assign_expert, (size_t)BK_max * sizeof(int));
    Malloc((void**)&t_d->state.assign_token, (size_t)BK_max * sizeof(int));
    Malloc((void**)&t_d->state.assign_weight, (size_t)BK_max * sizeof(float));
    Malloc((void**)&t_d->state.expert_counts, (size_t)nExp * sizeof(int));
    Malloc((void**)&t_d->state.expert_offsets, (size_t)(nExp + 1) * sizeof(int));
    Malloc((void**)&t_d->state.tokens_flat, (size_t)BK_max * sizeof(int));
    Malloc((void**)&t_d->state.weights_flat, (size_t)BK_max * sizeof(float));
    Malloc((void**)&t_d->state.x_by_expert, (size_t)BK_max * H * sizeof(float));
    Malloc((void**)&t_d->state.mlp1_by_expert, (size_t)BK_max * 2 * inter * sizeof(float));
    Malloc((void**)&t_d->state.gate_by_expert, (size_t)BK_max * inter * sizeof(float));
    Malloc((void**)&t_d->state.up_by_expert, (size_t)BK_max * inter * sizeof(float));
    Malloc((void**)&t_d->state.y_by_expert, (size_t)BK_max * H * sizeof(float));

    size_t S = (size_t)seq_len;
    std::vector<float> host_mask(S * S, 0.0f);
    if (S) {
        for (size_t i = 0; i < S; ++i) {
            for (size_t j = 0; j < S; ++j) {
                bool future = (j > i); // causal: mask positions beyond current
                bool sw_off = (conf->sliding_window > 0) && (i >= (size_t)conf->sliding_window) &&
                              (i - j >= (size_t)conf->sliding_window);
                if (future || sw_off)
                    host_mask[i * S + j] = -INFINITY;
            }
        }
        CHECK_HIP(hipMemcpy(t_d->state.mask, host_mask.data(), host_mask.size() * sizeof(float),
                            hipMemcpyHostToDevice));
    } else {
        // Shouldn't happen, but keep zeros if seq_len==0
        Z(t_d->state.mask, (size_t)seq_len * seq_len * sizeof(float));
    }

    // init zeros
    Z(t_d->state.x, (size_t)batch_size * H * sizeof(float));
    Z(t_d->state.t, (size_t)batch_size * H * sizeof(float));

    // Convert + copy host FP32 -> device FP16 (only slice we own)
    auto to_half = [](float v) { return __float2half(v); };
    auto copy_half = [&](__half* d, const float* h, size_t elems) {
        std::vector<__half> buf(elems);
        for (size_t i = 0; i < elems; ++i)
            buf[i] = to_half(h[i]);
        CHECK_HIP(hipMemcpy(d, buf.data(), elems * sizeof(__half), hipMemcpyHostToDevice));
    };

    for (int l = l_start; l < l_end; ++l) {
        int ll = l - l_start;

        copy_half(t_d->weights.rms_attn_w + (size_t)ll * H, W->rms_attn_w + (size_t)l * H, H);
        copy_half(t_d->weights.rms_ffn_w + (size_t)ll * H, W->rms_ffn_w + (size_t)l * H, H);

        copy_half(t_d->weights.w_qkv + (size_t)ll * qkv_perL_elems,
                  W->w_qkv + (size_t)l * qkv_perL_elems, qkv_perL_elems);
        copy_half(t_d->weights.w_o + (size_t)ll * wo_perL_elems, W->w_o + (size_t)l * wo_perL_elems,
                  wo_perL_elems);
        copy_half(t_d->weights.b_qkv + (size_t)ll * bqkv_perL_elems,
                  W->b_qkv + (size_t)l * bqkv_perL_elems, bqkv_perL_elems);
        copy_half(t_d->weights.b_o + (size_t)ll * bo_perL_elems, W->b_o + (size_t)l * bo_perL_elems,
                  bo_perL_elems);
        copy_half(t_d->weights.attn_sinks + (size_t)ll * sinks_perL,
                  W->attn_sinks + (size_t)l * sinks_perL, sinks_perL);

        // Router + MoE
        size_t router_elems = (size_t)H * nExp;
        size_t brouter_elems = (size_t)nExp;
        size_t mlp1_elems = (size_t)nExp * (2 * (size_t)inter) * H;
        size_t mlp2_elems = (size_t)nExp * (size_t)inter * H;
        size_t bmlp1_elems = (size_t)nExp * (2 * (size_t)inter);
        size_t bmlp2_elems = (size_t)nExp * (size_t)H;

        copy_half(t_d->weights.w_router + (size_t)ll * router_elems,
                  W->w_router + (size_t)l * router_elems, router_elems);
        copy_half(t_d->weights.b_router + (size_t)ll * brouter_elems,
                  W->b_router + (size_t)l * brouter_elems, brouter_elems);
        // copy_half(t_d->weights.w_mlp1 + (size_t)ll * mlp1_elems, W->w_mlp1 + (size_t)l *
        // mlp1_elems,
        //           mlp1_elems);
        // copy_half(t_d->weights.w_mlp2 + (size_t)ll * mlp2_elems, W->w_mlp2 + (size_t)l *
        // mlp2_elems,
        //           mlp2_elems);
        // copy_half(t_d->weights.b_mlp1 + (size_t)ll * bmlp1_elems,
        //           W->b_mlp1 + (size_t)l * bmlp1_elems, bmlp1_elems);
        // copy_half(t_d->weights.b_mlp2 + (size_t)ll * bmlp2_elems,
        //           W->b_mlp2 + (size_t)l * bmlp2_elems, bmlp2_elems);
        __half* dst_w_mlp1 = t_d->weights.w_mlp1 + (size_t)ll * mlp1_elems;
        __half* dst_w_mlp2 = t_d->weights.w_mlp2 + (size_t)ll * mlp2_elems;
        __half* dst_b_mlp1 = t_d->weights.b_mlp1 + (size_t)ll * bmlp1_elems;
        __half* dst_b_mlp2 = t_d->weights.b_mlp2 + (size_t)ll * bmlp2_elems;

        copy_large_tensor_streaming(&dst_w_mlp1, W->w_mlp1 + (size_t)l * mlp1_elems,
                                    mlp1_elems * sizeof(__half), "w_mlp1");
        copy_large_tensor_streaming(&dst_w_mlp2, W->w_mlp2 + (size_t)l * mlp2_elems,
                                    mlp2_elems * sizeof(__half), "w_mlp2");
        copy_large_tensor_streaming(&dst_b_mlp1, W->b_mlp1 + (size_t)l * bmlp1_elems,
                                    bmlp1_elems * sizeof(__half), "b_mlp1");
        copy_large_tensor_streaming(&dst_b_mlp2, W->b_mlp2 + (size_t)l * bmlp2_elems,
                                    bmlp2_elems * sizeof(__half), "b_mlp2");
    }

    if (has_embed) {
        copy_half(t_d->weights.token_embedding_table, W->token_embedding_table, (size_t)vocab * H);
    }
    if (has_out) {
        std::vector<__half> h(H);
        for (int i = 0; i < H; ++i)
            h[i] = to_half(W->rms_out_w[i]);
        CHECK_HIP(hipMemcpy(t_d->weights.rms_out_w, h.data(), (size_t)H * sizeof(__half),
                            hipMemcpyHostToDevice));

        size_t elems = (size_t)vocab * H;
        std::vector<__half> o(elems);
        for (size_t i = 0; i < elems; ++i)
            o[i] = to_half(W->out[i]);
        CHECK_HIP(
            hipMemcpy(t_d->weights.out, o.data(), elems * sizeof(__half), hipMemcpyHostToDevice));
    }

    t_d->pp = nullptr; // real shard: no manager
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    size_t used_mem = total_mem - free_mem;
    printf("✅ Hybrid precision model loaded: %.1f GB allocated\n",
           used_mem / (1024.0 * 1024.0 * 1024.0));
}

void copy_transformer_to_device_hybrid_grouped(OssTransformer* t_host, OssTransformerHybrid* t_root,
                                               int device_base, int pp_within_group) {
    int numDevs = 1;
    CHECK_HIP(hipGetDeviceCount(&numDevs));

    // Clamp pp to what fits from device_base onward.
    int pp = std::max(1, std::min(pp_within_group, numDevs - device_base));

    if (pp <= 1) {
        CHECK_HIP(hipSetDevice(device_base));
        copy_transformer_to_device_hybrid_pp(t_host, t_root, 0, t_host->config.n_layers,
                                             /*embed*/ true, /*out*/ true);
        t_root->pp = nullptr;
        return;
    }

    auto* mgr = new PPManager();
    mgr->pp = pp;
    mgr->hidden_dim = t_host->config.hidden_dim;
    mgr->n_layers = t_host->config.n_layers;
    mgr->vocab_size = t_host->config.vocab_size;
    mgr->batch_size = t_host->config.batch_size;

    auto cuts = partition_layers(mgr->n_layers, pp);
    mgr->shards.reserve(pp);
#pragma omp parallel for num_threads(pp) schedule(static)
    for (int p = 0; p < pp; ++p) {
        PPShard s{};
        s.device_id = device_base + p; // <-- KEY: offset into this DP group
        s.l_start = cuts[p].first;
        s.l_end = cuts[p].second;
        s.has_embed = (p == 0);
        s.has_out = (p == pp - 1);

        CHECK_HIP(hipSetDevice(s.device_id));
        s.shard = (OssTransformerHybrid*)std::malloc(sizeof(OssTransformerHybrid));
        std::memset(s.shard, 0, sizeof(OssTransformerHybrid));
        copy_transformer_to_device_hybrid_pp(t_host, s.shard, s.l_start, s.l_end, s.has_embed,
                                             s.has_out);
        mgr->shards[p] = s;
    }

    for (int i = 0; i < pp; i++)
        for (int j = 0; j < pp; j++)
            mgr->peer_ok[i][j] =
                (i == j) ? true : enable_p2p(mgr->shards[i].device_id, mgr->shards[j].device_id);

    t_root->pp = mgr;
    std::memset(&t_root->weights, 0, sizeof(t_root->weights));
    std::memset(&t_root->state, 0, sizeof(t_root->state));
    t_root->config = t_host->config;
}

void free_transformer_on_device_hybrid_single(OssTransformerHybrid* t_d) {
    printf("\033[1;92m==================================================================\033["
           "0m\n\033[1;92m♻️  FREE GPU MEMORY...\033[0m\n");

    // Free FP16 weights
    CHECK_HIP(hipFree(t_d->weights.token_embedding_table));
    CHECK_HIP(hipFree(t_d->weights.rms_attn_w));
    CHECK_HIP(hipFree(t_d->weights.rms_ffn_w));
    CHECK_HIP(hipFree(t_d->weights.w_qkv));
    CHECK_HIP(hipFree(t_d->weights.w_o));
    CHECK_HIP(hipFree(t_d->weights.b_qkv));
    CHECK_HIP(hipFree(t_d->weights.b_o));
    CHECK_HIP(hipFree(t_d->weights.attn_sinks));
    CHECK_HIP(hipFree(t_d->weights.w_router));
    CHECK_HIP(hipFree(t_d->weights.b_router));
    CHECK_HIP(hipFree(t_d->weights.w_mlp1));
    CHECK_HIP(hipFree(t_d->weights.w_mlp2));
    CHECK_HIP(hipFree(t_d->weights.b_mlp1));
    CHECK_HIP(hipFree(t_d->weights.b_mlp2));
    CHECK_HIP(hipFree(t_d->weights.rms_out_w));
    CHECK_HIP(hipFree(t_d->weights.out));

    // Free FP32 state buffers
    CHECK_HIP(hipFree(t_d->state.x));
    CHECK_HIP(hipFree(t_d->state.t));
    CHECK_HIP(hipFree(t_d->state.tb));
    CHECK_HIP(hipFree(t_d->state.tb2));
    CHECK_HIP(hipFree(t_d->state.router_score));
    CHECK_HIP(hipFree(t_d->state.topk_v));
    CHECK_HIP(hipFree(t_d->state.topk_i));
    CHECK_HIP(hipFree(t_d->state.mlp1_out));
    CHECK_HIP(hipFree(t_d->state.gate));
    CHECK_HIP(hipFree(t_d->state.up));
    CHECK_HIP(hipFree(t_d->state.gate_up));
    CHECK_HIP(hipFree(t_d->state.e_agg));
    CHECK_HIP(hipFree(t_d->state.qkv));
    CHECK_HIP(hipFree(t_d->state.q));
    CHECK_HIP(hipFree(t_d->state.att));
    CHECK_HIP(hipFree(t_d->state.logits));
    CHECK_HIP(hipFree(t_d->state.key_cache));
    CHECK_HIP(hipFree(t_d->state.value_cache));
    CHECK_HIP(hipFree(t_d->state.mask));
    CHECK_HIP(hipFree(t_d->state.d_batch_indices));
    CHECK_HIP(hipFree(t_d->state.d_tokens));
    CHECK_HIP(hipFree(t_d->state.d_pos_per_token));
    CHECK_HIP(hipFree(t_d->state.cos_vals));
    CHECK_HIP(hipFree(t_d->state.sin_vals));

    CHECK_HIP(hipFree(t_d->state.assign_expert));
    CHECK_HIP(hipFree(t_d->state.assign_token));
    CHECK_HIP(hipFree(t_d->state.assign_weight));
    CHECK_HIP(hipFree(t_d->state.expert_counts));
    CHECK_HIP(hipFree(t_d->state.expert_offsets));
    CHECK_HIP(hipFree(t_d->state.tokens_flat));
    CHECK_HIP(hipFree(t_d->state.weights_flat));
    CHECK_HIP(hipFree(t_d->state.x_by_expert));
    CHECK_HIP(hipFree(t_d->state.mlp1_by_expert));
    CHECK_HIP(hipFree(t_d->state.gate_by_expert));
    CHECK_HIP(hipFree(t_d->state.up_by_expert));
    CHECK_HIP(hipFree(t_d->state.y_by_expert));

    size_t free_mem, total_mem;
    CHECK_HIP(hipMemGetInfo(&free_mem, &total_mem));
    printf("GPU memory freed: %.1f GB free / %.1f GB total\n",
           free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));
}

void free_transformer_on_device_hybrid(OssTransformerHybrid* t_root) {
    if (!t_root)
        return;
    if (t_root->pp == nullptr) {
        // original single-GPU free (keep your existing body)
        // ...
        // Example:
        // if (t_root->weights.token_embedding_table) hipFree(...);
        // ...
        free_transformer_on_device_hybrid_single(t_root);
        return;
    }
    // PP free
    PPManager* mgr = t_root->pp;
    for (auto& s : mgr->shards) {
        CHECK_HIP(hipSetDevice(s.device_id));
        free_transformer_on_device_hybrid_single(
            s.shard); // this calls single-GPU path of the shard
        std::free(s.shard);
    }
    if (mgr->host_bounce)
        CHECK_HIP(hipHostFree(mgr->host_bounce));
    delete mgr;
    t_root->pp = nullptr;
    std::memset(t_root, 0, sizeof(*t_root));
}
