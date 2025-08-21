<div align="center">

# gpt-oss inference engine

<img width="1589" height="734" alt="image" src="https://github.com/user-attachments/assets/8a797e2b-6ae5-4383-b6ff-4d5b914bbece" />

</div>

## abstract

This repository implements an inference serving system for **gpt-oss** models (20B & 120B) using a minimal C/C++ runtime derived from `llama2.c`. It targets **single-node, multi-GPU** execution on AMD MI250 with custom HIP kernels, and supports CPU execution with OpenMP/pthreads.

- **Baseline:** CPU-only C/C++ code (single-prompt, greedy decoding)
- **Extension:** HIP GPU execution and multi-GPU parallelism on a single node

---

### goals

- ✅ **Correctness first** — keep simple checks/metrics to verify output validity.
- 🚀 **Throughput** — maximize tokens/sec via CPU threading + HIP GPU kernels.
- 📚 **Scope** — single node, multi-GPU execution for 20B and 120B models.

---

## resources

- **Model binaries:**
  - `/nfs/gpu_trainee/final-project/modelbin/`
    - `gpt-oss-7m.bin` (debug only)
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
srun --gres=gpu:<N> ./run /path/to/model.bin -m generate -i "..."
```

> The training cluster provides up to **4 nodes** for experimentation, but the **project deliverable focuses on single-node, multi-GPU execution**.

---

## build & run

```bash
chmod +x run.sh
ln -s run.sh run
```

### build

```bash
./runh build [default|fast|omp]
```

## run

**Chat mode**

```bash
./run run -m chat
```

**Single-prompt generate**

```bash
./run run -m generate -i "Write a haiku about parallelism."
```

**Batch evaluation**

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

## rules

- **Do not modify:** `run.cpp`, `getp_eval.cpp`, `Makefile`.
- **Implement all GPU kernels from scratch** — **no external GPU libraries**.
- **Target:** single-node, multi-GPU execution.

---

## refs

- [GPT-OSS](https://openai.com/index/introducing-gpt-oss/)
- [llama2.c](https://github.com/karpathy/llama2.c)
- [AMD ROCm](https://rocm.docs.amd.com/)
- [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [OpenMP](https://www.openmp.org/specifications/)
- [Slurm](https://slurm.schedmd.com/documentation.html)
- [tiktoken](https://github.com/openai/tiktoken)
