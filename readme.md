<div align="center">

# Implementing GPT-OSS from Scratch on AMD GPUs

 <p>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/github/actions/workflow/status/tuanlda78202/gpt-oss-amd/ci.yaml?branch=main&label=CI&logo=github" alt="CI Status">
    <img src="https://img.shields.io/github/last-commit/tuanlda78202/gpt-oss-amd?&label=commit" alt="Last Commit">
 </p>

<img width="1589" height="734" alt="image" src="https://github.com/user-attachments/assets/8a797e2b-6ae5-4383-b6ff-4d5b914bbece" />

</div>

## abstract

After six years-the first time since GPT-2, OpenAI has released new open-weight LLMs, `gpt-oss-20b` and `gpt-oss-120b`. From day one, many inference engines such as llama.cpp, vLLM, and SGLang have supported these models; however, most focus on maximizing throughput using CUDA for NVIDIA GPUs, offering limited support for AMD GPUs. Moreover, their library-oriented implementations are often complex and difficult to adapt for personal/experimental use cases.

To address these limitations, we present `gpt-oss-amd`, a pure C++ implementation of OpenAI’s GPT-OSS models designed to **maximize inference throughput on AMD GPUs without relying on external libraries**. Our goal is to explore end-to-end LLM optimization, from kernel-level improvements to system-level design, providing insights for researchers and developers interested in high-performance computing and model-level optimization.

Inspired by [llama2.c](https://github.com/karpathy/llama2.c), our implementation uses HIP (an AMD equivalent to CUDA) and avoids dependencies such as rocBLAS, hipBLAS, RCCL, and MPI. We employ a range of optimization techniques for both the 20B and 120B models, including efficient model loading, batching, multi-streaming, multi-GPUs communication, optimized CPU–GPU–SRAM memory access, FlashAttention, matrix-core–based GEMM, and load balancing in MoE routing. Experiments on a single node with 8× AMD MI250 GPUs show that our implementation achieves over 30k TPS on the 20B model and 10k TPS on the 120B model in custom benchmarks, demonstrating the effectiveness of our optimizations and the strong potential of AMD GPUs for large-scale LLM inference.

---

## structure

```plain
├── include/              # Header files
├── src/
│   ├── getp/             # Request serving and runtime logic
│   ├── hip/              # Custom HIP kernels for AMD GPUs
│   ├── forward.cpp       # Model forward pass implementation
├── tests/                # Evaluation scripts
├── tools/                # Model/tokenizer conversion and HF inference utilities
└── run.sh                # Build and run script
```

### goals

- **Correctness** — keep simple checks/metrics to verify output validity.
- **Throughput** — maximize tokens/sec via CPU threading + HIP GPU kernels.
- **Scope** — single node, multi-GPU execution for 20B and 120B models.

---

## resources

- **Model binaries:**
  - `gpt-oss-20b.bin`
  - `gpt-oss-120b.bin`

- **Tokenizer:** compatible with OpenAI **`o200k_harmony`** (via `tiktoken`).

### env

- **Hardware:** single node with up to **8× AMD MI250 GPUs**.
- **GPU:** HIP/ROCm (write **all GPU kernels from scratch**, no GPU libs).
- **CPU:** GCC/Clang with **OpenMP**/**pthreads**.
- **OS:** Slurm for job execution.

### slurm cluster

Login node:

```bash
ssh getp<XX>@203.205.18.240
```

Compute node:

```bash
# srun --gres=gpu:1 rocm-smi
srun --gres=gpu:<N> ./run /path/to/model.bin -m generate -i "..."
```

> The training cluster provides up to **4 nodes** for experimentation, but the **project deliverable focuses on single-node, multi-GPU execution**.

---

## build & run

```bash
uv sync
source .venv/bin/activate
pre-commit install

chmod +x run.sh
ln -s run.sh run
```

### build

```bash
./run build [default|fast|omp]
```

## run

### chat

```bash
./run run -m chat
```

### single-prompt

```bash
./run run -m generate -i "Write a haiku about parallelism."
```

### batch

```bash
./run run -m getp
```

---

## eval

| Mode       | Description                             | Example                                                       |
| ---------- | --------------------------------------- | ------------------------------------------------------------- |
| `chat`     | Interactive turn-based generation       | `./run.sh -c model.bin -m chat`                               |
| `generate` | Single prompt → completion              | `./run.sh -c model.bin -m generate -i "..."`                  |
| `getp`     | Multi-prompt batch for final evaluation | `./run.sh -c model.bin -m getp -i prompts.txt -o outputs.txt` |

- Maintain **correctness metrics** (e.g., checksum/sanity prompts).
- Report **tokens/sec** for each mode and model size.
- Optimize across:
  - CPU threading (OpenMP/pthreads)
  - HIP kernels (coalescing, tiling, occupancy)
  - Multi-GPU parallelization (pipeline/tensor-level)

---

## refs

- [GPT-OSS](https://openai.com/index/introducing-gpt-oss/)
- [llama2.c](https://github.com/karpathy/llama2.c)
- [AMD ROCm](https://rocm.docs.amd.com/)
- [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [OpenMP](https://www.openmp.org/specifications/)
- [Slurm](https://slurm.schedmd.com/documentation.html)
