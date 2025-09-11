#include <cstdio>
#include <cstring>

bool g_enable_profiling = false;
void set_profiling_enabled(bool enabled) { g_enable_profiling = enabled; }

struct BatchForwardTimings {
    double setup_time;
    double embedding_time;
    double layer_rmsnorm_time;
    double qkv_projection_time;
    double split_qkv_time;
    double rope_time;
    double attention_time;
    double output_projection_time;
    double residual_time;
    double mlp_rmsnorm_time;
    double moe_routing_time;
    double expert_processing_time;

    double moe_setup_time;
    double moe_assignment_time;
    double moe_scan_time;
    double moe_compact_time;
    double moe_gather_time;
    double moe_queue_build_time;
    double moe_mlp1_time;
    double moe_swiglu_time;
    double moe_mlp2_time;
    double moe_scatter_time;
    double moe_sync_time;

    double final_ops_time;
    double total_time;

    int num_calls;
    int total_layers;

    BatchForwardTimings() { reset(); }

    void reset() {
        setup_time = embedding_time = layer_rmsnorm_time = qkv_projection_time = 0.0;
        split_qkv_time = rope_time = attention_time = output_projection_time = 0.0;
        residual_time = mlp_rmsnorm_time = moe_routing_time = expert_processing_time = 0.0;

        moe_setup_time = moe_assignment_time = moe_scan_time = moe_compact_time = 0.0;
        moe_gather_time = moe_queue_build_time = moe_mlp1_time = moe_swiglu_time = 0.0;
        moe_mlp2_time = moe_scatter_time = moe_sync_time = 0.0;

        final_ops_time = total_time = 0.0;
        num_calls = total_layers = 0;
    }

    void print_summary() {
        if (num_calls == 0)
            return;

        printf("%.*s\n", 70,
               "======================================================================");
        printf("BATCH FORWARD TIMING SUMMARY (%d calls, %d total layers)\n", num_calls,
               total_layers);
        printf("%.*s\n", 70,
               "======================================================================");

        // Main sections
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Setup & Memory", setup_time / num_calls,
               100.0 * setup_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Embedding", embedding_time / num_calls,
               100.0 * embedding_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Layer RMSNorm", layer_rmsnorm_time / num_calls,
               100.0 * layer_rmsnorm_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "QKV Projection", qkv_projection_time / num_calls,
               100.0 * qkv_projection_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Split QKV", split_qkv_time / num_calls,
               100.0 * split_qkv_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "RoPE", rope_time / num_calls,
               100.0 * rope_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Attention", attention_time / num_calls,
               100.0 * attention_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Output Projection",
               output_projection_time / num_calls, 100.0 * output_projection_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Residual", residual_time / num_calls,
               100.0 * residual_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "MLP RMSNorm", mlp_rmsnorm_time / num_calls,
               100.0 * mlp_rmsnorm_time / total_time);
        printf("%-25s: %8.3f ms (%.1f%%)\n", "MoE Routing", moe_routing_time / num_calls,
               100.0 * moe_routing_time / total_time);

        // Expert Processing total
        printf("%-25s: %8.3f ms (%.1f%%)\n", "Expert Processing [TOTAL]",
               expert_processing_time / num_calls, 100.0 * expert_processing_time / total_time);

        // Detailed MoE Expert Processing Breakdown
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "MoE Setup/Memset", moe_setup_time / num_calls,
               100.0 * moe_setup_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Assignment/Counting",
               moe_assignment_time / num_calls, 100.0 * moe_assignment_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Exclusive Scan", moe_scan_time / num_calls,
               100.0 * moe_scan_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Compact by Expert", moe_compact_time / num_calls,
               100.0 * moe_compact_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Gather Inputs", moe_gather_time / num_calls,
               100.0 * moe_gather_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "Queue Build/Meta",
               moe_queue_build_time / num_calls, 100.0 * moe_queue_build_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "MLP1 MatVec", moe_mlp1_time / num_calls,
               100.0 * moe_mlp1_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "SwiGLU Activation", moe_swiglu_time / num_calls,
               100.0 * moe_swiglu_time / total_time);
        printf("  ├─ %-21s: %8.3f ms (%.1f%%)\n", "MLP2 MatVec", moe_mlp2_time / num_calls,
               100.0 * moe_mlp2_time / total_time);
        printf("  └─ %-21s: %8.3f ms (%.1f%%)\n", "Scale/Scatter", moe_scatter_time / num_calls,
               100.0 * moe_scatter_time / total_time);

        printf("%-25s: %8.3f ms (%.1f%%)\n", "Final Operations", final_ops_time / num_calls,
               100.0 * final_ops_time / total_time);

        printf("%.*s\n", 70,
               "======================================================================");
        printf("%-25s: %8.3f ms\n", "TOTAL", total_time / num_calls);
        printf("%.*s\n", 70,
               "======================================================================");
        fflush(stdout);
    }
};

BatchForwardTimings g_batch_timings;

void reset_batch_timings() {
    if (g_enable_profiling) {
        g_batch_timings.reset();
    }
}

void print_batch_timing_summary() {
    if (g_enable_profiling) {
        g_batch_timings.print_summary();
    }
}
