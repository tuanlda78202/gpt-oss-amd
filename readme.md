<div align="center">

# gpt-oss inference engine

<img width="1589" height="734" alt="image" src="https://github.com/user-attachments/assets/8a797e2b-6ae5-4383-b6ff-4d5b914bbece" />

</div>

## abstract

This repository implements an inference serving system for **gpt-oss** models (20B & 120B parameters) using a minimal C/C++ runtime derived from **`llama2.c`**. It targets **single-node, multi-GPU** execution on AMD MI250 with custom HIP kernels, and supports CPU execution with OpenMP/pthreads.

- **Baseline:** CPU-only C/C++ code (single-prompt, greedy decoding).
- **Extension:** HIP GPU execution and multi-GPU parallelism on a single node.
- **Weights:** Pre-exported binary weights (e.g., from Hugging Face checkpoints).
- **Tokenizer:** OpenAI `o200k_harmony`-compatible.

---

### goals

- ✅ **Correctness first** — keep simple checks/metrics to verify output validity.
- 🚀 **Throughput** — maximize tokens/sec via CPU threading + HIP GPU kernels.
- 🧱 **Scope** — single node, multi-GPU execution for 20B and 120B models.

---

## tree

```
.
├─ run.cpp               # DO NOT MODIFY (entry for single-prompt)
├─ run_eval.cpp          # DO NOT MODIFY (evaluation runner)
├─ Makefile              # DO NOT MODIFY
├─ run_exec.cpp          # Add new execution paths / orchestrations here
├─ include/              # Common headers
├─ cpu/                  # CPU ops (OpenMP / pthreads)
├─ hip/                  # HIP GPU kernels (implement from scratch)
└─ tools/                # Utility scripts (optional)
```

> **Do not** edit `run.cpp`, `run_eval.cpp`, or `Makefile`. Add new logic in `run_exec.cpp` and new source files as needed.

---

## resources

- **Model binaries (internal/shared path):**

  - `/nfs/gpu_trainee/final-project/modelbin/`

    - `gpt-oss-7m.bin` (debug only)
    - `gpt-oss-20b.bin`
    - `gpt-oss-120b.bin`

- **Tokenizer:** compatible with OpenAI **`o200k_harmony`** (via `tiktoken`).

---

### environment

- **Hardware:** single node with up to **8× AMD MI250 GPUs**.
- **GPU toolchain:** HIP/ROCm (write **all GPU kernels from scratch**, no GPU libs).
- **CPU toolchain:** GCC/Clang with **OpenMP** and/or **pthreads**.
- **OS/Cluster:** Slurm for job execution.

### slurm cluster

Login:

```bash
ssh getp<XX>@203.205.18.240
```

Run jobs with GPUs:

```bash
# Request N GPUs (default: 1)
srun --gres=gpu:<N> ./run /path/to/model.bin -m generate -i "..."
```

> The training cluster provides up to **4 nodes** for experimentation, but the **project deliverable focuses on single-node, multi-GPU execution**.

---

## build & run

- [run docs](docs/setup.md)

```bash
export MODELS_ROOT="/nfs/gpu_trainee/final-project/models"
export MODELBIN_ROOT="/nfs/gpu_trainee/final-project/modelbin"
```

### build

```bash
make run      # Default compilation, very slow
make runfast  # Compiled with -O3 optimization
make runomp   # Compiled with -O3 and -fopenmp
```

> If you add new source files (e.g., HIP kernels and orchestration in `run_exec.cpp`), keep the Makefile unchanged by including new code through headers or `run_exec.cpp`.

---

## run

**Chat mode**

```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m chat
```

**Single-prompt generate**

```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m generate -i "Write a haiku about parallelism."
```

**Batch evaluation (“getp”)**

```bash
./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m getp -i data/input.txt -o data/output.txt
```

> Your extended paths (e.g., GPU/multi-GPU) should be wired through `run_exec.cpp` and invoked by flags you define (keep CLI consistent with `run.cpp` semantics).

---

## evaluation

| Mode       | Description                             | Example                                                 |
| ---------- | --------------------------------------- | ------------------------------------------------------- |
| `chat`     | Interactive turn-based generation       | `./run model.bin -m chat`                               |
| `generate` | Single prompt → completion              | `./run model.bin -m generate -i "..."`                  |
| `getp`     | Multi-prompt batch for final evaluation | `./run model.bin -m getp -i prompts.txt -o outputs.txt` |

### performance & correctness

- Maintain **correctness metrics** (e.g., checksum/sanity prompts).
- Report **tokens/sec** for each mode and model size.
- Optimize across:

  - CPU threading (OpenMP/pthreads)
  - HIP kernels (coalescing, tiling, occupancy)
  - Multi-GPU parallelization (pipeline/tensor-level where appropriate)

---

## rules

- **Do not modify:** `run.cpp`, `run_eval.cpp`, `Makefile`.
- **Implement all GPU kernels from scratch** — **no external GPU libraries**.
- **Target:** single-node, multi-GPU execution.
- Keep tokenizer compatibility (`o200k_harmony`) and weight I/O format intact.

---

## refs

- [GPT-OSS](https://openai.com/index/introducing-gpt-oss/)
- [llama2.c](https://github.com/karpathy/llama2.c)
- [AMD ROCm](https://rocm.docs.amd.com/)
- [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [OpenMP](https://www.openmp.org/specifications/)
- [Slurm](https://slurm.schedmd.com/documentation.html)
- [tiktoken](https://github.com/openai/tiktoken)

---

## license

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
