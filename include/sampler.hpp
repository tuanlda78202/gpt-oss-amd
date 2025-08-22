#pragma once

typedef struct {
    float prob;
    int index;
} OssProbIndex; // struct used when sorting probabilities in top-p sampling

typedef struct {
    int vocab_size;
    OssProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} OssSampler;

int sample_argmax_oss(float* probabilities, int n);

int sample_mult_oss(float* probabilities, int n, float coin);

int compare_oss(const void* a, const void* b);

int sample_topp_oss(float* probabilities, int n, float topp, OssProbIndex* probindex, float coin);

void build_sampler_oss(OssSampler* sampler, int vocab_size, float temperature, float topp,
                       unsigned long long rng_seed);

void free_sampler_oss(OssSampler* sampler);

int sample_oss(OssSampler* sampler, float* logits);
