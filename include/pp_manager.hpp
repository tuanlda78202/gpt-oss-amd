#pragma once
#include "../include/model.hpp"
#include <vector>

struct PPShard {
    int device_id = 0;
    int l_start = 0; // inclusive
    int l_end = 0;   // exclusive
    bool has_embed = false;
    bool has_out = false;
    OssTransformerHybrid* shard; // per-GPU shard
};

struct PPManager {
    int pp = 1;
    int hidden_dim = 0;
    int n_layers = 0;
    int vocab_size = 0;
    std::vector<PPShard> shards;
    bool peer_ok[16][16] = {};    // P2P reachability matrix
    float* host_bounce = nullptr; // pinned bounce buffer when no P2P
};
