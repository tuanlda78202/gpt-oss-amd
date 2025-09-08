#include "../include/sampler.hpp"
#include "hip/sampling.hip"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

int sample_argmax_oss(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult_oss(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare_oss(const void* a, const void* b) {
    OssProbIndex* a_ = (OssProbIndex*)a;
    OssProbIndex* b_ = (OssProbIndex*)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

int sample_topp_oss(float* probabilities, int n, float topp, OssProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(OssProbIndex), compare_oss);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler_oss(OssSampler* sampler, int vocab_size, float temperature, float topp,
                       unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex =
        reinterpret_cast<OssProbIndex*>(malloc(sampler->vocab_size * sizeof(OssProbIndex)));
}

void free_sampler_oss(OssSampler* sampler) { free(sampler->probindex); }

int sample_oss(OssSampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax_oss(logits, sampler->vocab_size);
    } else {
    }
    return next;
}

int sample_oss_gpu(OssSampler* sampler, float* logits_d) {
    thread_local int* result_d = nullptr;
    if (result_d == nullptr) {
        CHECK_HIP(hipMalloc(&result_d, sizeof(int)));
    }

    if (sampler->temperature == 0.0f) {
        sample_argmax_hip_device(logits_d, sampler->vocab_size, result_d);
    } else {
        sample_multinomial_hip_device(logits_d, sampler->vocab_size, sampler->temperature,
                                      sampler->rng_state, result_d);
        sampler->rng_state = (sampler->rng_state * 1103515245 + 12345) & 0x7fffffff;
    }

    int next;
    CHECK_HIP(hipMemcpy(&next, result_d, sizeof(int), hipMemcpyDeviceToHost));
    return next;
}
