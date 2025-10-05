"""
Evaluation script for comparing submission completions against reference completions using METEOR and BERTScore F1 metrics.

Usage:
    - Pass --model_type (20b or 120b) to automatically select submission and reference files.
    - Or pass --submission and --references to specify custom files.

Thresholds:
    - Reads threshold values for METEOR and BERTScore F1 from threshold.json in the evaluation folder.
    - Asserts at the end that both metrics meet or exceed their respective thresholds.
    - Raises an AssertionError if either metric is below its threshold.

Outputs:
    - Prints METEOR and BERTScore F1 scores, and other evaluation details.
"""

import argparse
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import nltk
import tiktoken
from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_type",
        choices=["20b", "120b"],
        help="Model type: 20b or 120b. If provided, sets default submission and references paths.",
    )
    parser.add_argument(
        "-s",
        "--submission",
        type=pathlib.Path,
        help="Submission completions file (space-separated token IDs per line)",
    )
    parser.add_argument(
        "-r",
        "--references",
        type=pathlib.Path,
        help="Reference completions file (space-separated token IDs per line)",
    )
    parser.add_argument(
        "-e",
        "--encoding",
        default="o200k_harmony",
        help="Tiktoken encoding name",
    )
    args = parser.parse_args()
    return args


def ensure_nltk_resources():
    print("ensuring nltk resources...", flush=True)
    for r in ["punkt", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{r}") if r == "punkt" else nltk.data.find(
                f"corpora/{r}"
            )
            print(f"nltk resource available: {r}", flush=True)
        except LookupError:
            print(f"downloading nltk resource: {r}", flush=True)
            nltk.download(r, quiet=True)
            print(f"downloaded: {r}", flush=True)


def read_completions(path: pathlib.Path, enc) -> List[str]:
    print(f"decoding file: {path}", flush=True)
    comps: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                comps.append("")
                continue
            ids = [int(x) for x in line.split()]
            comps.append(enc.decode(ids))
    print(f"decoded lines: {len(comps)} from {path}", flush=True)
    return comps


def _meteor_for_pair(pair: Tuple[List[str], List[str]]) -> float:
    refs_tokens, hyp_tokens = pair
    return float(meteor_score([refs_tokens], hyp_tokens))


def compute_meteor(refs: List[str], subm: List[str]) -> tuple[int, float]:
    ensure_nltk_resources()
    n = min(len(refs), len(subm))
    print(f"aligning by index, items: {n}", flush=True)
    if n == 0:
        return 0, 0.0

    refs_cut = refs[:n]
    subm_cut = subm[:n]

    pairs = ((r.split(), s.split()) for r, s in zip(refs_cut, subm_cut))
    max_workers = os.cpu_count() or 1
    print(f"computing METEOR in parallel with {max_workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        scores = list(tqdm(ex.map(_meteor_for_pair, pairs), total=n))

    avg = float(sum(scores) / n)
    return n, avg


def _infer_num_layers(bs_model_type: str) -> int:
    name = Path(bs_model_type).name.lower()
    if "xxlarge" in name:
        return 48
    if "xlarge" in name:
        return 24
    if "large" in name:
        return 24
    if "base" in name:
        return 12
    return 24


def compute_bertscore(
    refs: List[str], subm: List[str], bs_model_type: str
) -> tuple[int, float]:
    n = min(len(refs), len(subm))
    if n == 0:
        return 0, 0.0
    print(f"loading BERTScore model: {bs_model_type}...", flush=True)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        print(
            f"BERTScore device: GPU ({torch.cuda.device_count()} device(s))", flush=True
        )
    else:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            torch.set_num_threads(os.cpu_count() or 1)
        except Exception:
            pass
        print("BERTScore device: CPU", flush=True)

    num_layers = _infer_num_layers(bs_model_type)
    scorer = BERTScorer(
        lang="en", model_type=bs_model_type, num_layers=num_layers, device=device
    )
    total_sum = 0.0
    total_cnt = 0
    bs_batch = 64
    chunk_size = 128
    with tqdm(total=n, desc="BERTScore", dynamic_ncols=True, mininterval=0.2) as pbar:
        for i in range(0, n, chunk_size):
            j = min(i + chunk_size, n)
            _, _, F1 = scorer.score(
                subm[i:j], refs[i:j], batch_size=bs_batch, verbose=True
            )
            total_sum += float(F1.sum().item())
            total_cnt += j - i
            pbar.update(j - i)

    return n, float(total_sum / max(1, total_cnt))


def ensure_bs_model(local_dir: Path) -> Path:
    local_dir = Path(local_dir)
    need_download = True
    if local_dir.exists():
        needed = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
        ]
        if all((local_dir / f).exists() for f in needed):
            need_download = False
    if need_download:
        print(f"downloading BERTScore model to {local_dir} ...", flush=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            os.environ.setdefault("HF_HUB_ENABLE_XET", "0")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=local_dir,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download {local_dir}: {e}")
    else:
        print(f"found local BERTScore model at {local_dir}", flush=True)
    return local_dir


def main():
    import json

    threshold_path = Path(__file__).parent / "threshold.json"
    with open(threshold_path, "r") as f:
        thresholds = json.load(f)

    print("parsing args...", flush=True)
    args = parse_args()

    # Determine submission and references paths
    if args.model_type:
        subm_path = Path(f"./submission/output_{args.model_type}_token_ids.txt")
        refs_path = Path(f"./references/output_{args.model_type}_token_ids.txt")
    else:
        if not args.submission or not args.references:
            print(
                "Error: If --model_type is not provided, you must pass both --submission and --references.",
                flush=True,
            )
            exit(1)
        subm_path = args.submission
        refs_path = args.references

    print(f"loading tiktoken encoding: {args.encoding}", flush=True)
    enc = tiktoken.get_encoding(args.encoding)
    print(f"reading submission: {subm_path}", flush=True)
    subm = read_completions(subm_path, enc)
    print(f"reading references: {refs_path}", flush=True)
    refs = read_completions(refs_path, enc)

    print("computing METEOR...", flush=True)
    n, meteor = compute_meteor(refs, subm)
    print(f"meteor\t{round(meteor, 6)}")

    print("computing BERTScore (roberta-large-mnli)...", flush=True)
    models_root = os.environ.get("MODELS_ROOT")
    bs_model_root = Path(models_root) / "bs_model" / "roberta-large-mnli"
    roberta_local = ensure_bs_model(bs_model_root)
    n_b, bsf1 = compute_bertscore(refs, subm, str(roberta_local))
    print(f"bertscore_f1\t{round(bsf1, 6)}")

    print("=" * 10, "\n")
    print("done.", flush=True)
    print(f"items\t{n}")
    print(f"meteor\t{round(meteor, 6)}")
    print(f"bertscore_f1\t{round(bsf1, 6)}")

    # Assert thresholds
    assert meteor >= thresholds["meteor"], (
        f"METEOR {meteor} < threshold {thresholds['meteor']}"
    )
    assert bsf1 >= thresholds["bertscore_f1"], (
        f"BERTScore F1 {bsf1} < threshold {thresholds['bertscore_f1']}"
    )


if __name__ == "__main__":
    main()
