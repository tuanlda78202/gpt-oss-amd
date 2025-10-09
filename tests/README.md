# Evaluation

Evaluate generated completions against references using **METEOR** and **BERTScore (F1)**.

## Prerequisites

- Python 3.10+
- Installed packages: `tiktoken`, `nltk`, `bert-score`, `tqdm`
- **GPU strongly recommended**. On AMD GPUs, ensure PyTorch is installed **with ROCm**. CPU will be very slow.

> NLTK resources (`punkt`, `wordnet`, `omw-1.4`) are downloaded automatically on first run. If your machine is offline, pre-download them:
>
> ```bash
> python -c "import nltk; [nltk.download(x) for x in ['punkt','wordnet','omw-1.4']]"
> ```

## Quick start (SLURM)

Evaluate the **20B** model outputs on 2 GPUs:

```bash
srun --gres=gpu:2 python eval.py -m 20b
```

Evaluate the **120B** model:

```bash
srun --gres=gpu:2 python eval.py -m 120b
```

## CLI options

`eval.py` supports:

- `-m, --model_type {20b,120b}`
  Selects default input paths for submission and references.
- `-s, --submission PATH`
  Override path to submission completions (space-separated token IDs per line).
- `-r, --references PATH`
  Override path to reference completions (space-separated token IDs per line).
- `-e, --encoding NAME` (default: `o200k_harmony`)
  Tiktoken encoding used to decode token IDs.

Run `python eval.py -h` for the full help text. See the docstring in `eval.py` for details on defaults and behavior.

> If `threshold.json` is present in the evaluation folder, the script reads target thresholds for METEOR and BERTScore F1 and **raises an `AssertionError`** if either metric is below its threshold.

## Sample output (truncated)

```bash
20b
parsing args...
loading tiktoken encoding: o200k_harmony
reading submission: submission/output_20b_token_ids.txt
decoded lines: 4096 from submission/output_20b_token_ids.txt
reading references: references/output_20b_token_ids.txt
decoded lines: 4096 from references/output_20b_token_ids.txt

computing METEOR...
ensuring nltk resources...
aligning by index, items: 4096
computing METEOR in parallel with 96 workers...
100%|██████████| 4096/4096 [00:46<00:00, 88.0it/s]
meteor  0.384259

computing BERTScore (roberta-large-mnli)...
BERTScore device: GPU (2 device(s))
BERTScore: 100%|██████████| 4096/4096 [02:22<00:00, 28.7it/s]
bertscore_f1    0.966357

==========
done.
items           4096
meteor          0.384259
bertscore_f1    0.966357
```

```bash
120b
==========
done.
items           4096
meteor          0.404538
bertscore_f1    0.96955
```

**Expected runtime:** \~4 minutes for 4096 samples on 2 GPUs (your hardware and load may vary).

## Tips & troubleshooting

- **GPU vs CPU:** If logs show CPU, performance will be much slower. Verify:

  ```bash
  ROCm, ensure `torch.version.hip` is not `None`.
  ```
