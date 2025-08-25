"""
Contest Evaluation Script for LLM Generation-Only Outputs
- Primary metrics: BERTScore (F1), METEOR
- Auxiliary: distinct-1/2, repetition rate, average length
- Modes:
  * Reference-based: requires --refs, computes METEOR + BERTScore vs gold
  * Reference-free: pairwise symmetric BERTScore among candidates
- Fairness: evaluate on the intersection of IDs across all candidates (and refs if provided).
"""

import argparse
import hashlib
import json
import pathlib
import statistics
from typing import Dict, List

import evaluate
import nltk
import pandas as pd
import tiktoken
from jsonschema import ValidationError, validate


def parseCLI():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompts",
        default="../data/input.txt",
        type=pathlib.Path,
        help="Path to prompts file",
    )
    parser.add_argument(
        "-s",
        "--submission",
        default="../data/output.txt",
        type=pathlib.Path,
        help="Path to trainee's completion file",
    )
    # this json file is assumed to have the form {"id", "prompt", "completion"}
    parser.add_argument(
        "-r",
        "--references",
        type=pathlib.Path,
        help="Path to reference completion file used for evaluation",
    )
    args = parser.parse_args()
    return args


def validate_references(refs: List[Dict], schema_path: pathlib.Path) -> bool:
    with open(schema_path, "r") as f:
        schema = json.load(f)

    for i, entry in enumerate(refs, start=1):
        try:
            validate(instance=entry, schema=schema)
        except ValidationError as e:
            print(f"[SCHEMA ERROR] Line {i}: {e.message}")
            return False
        print("[SCHEMA CHECK] All references match schema")
        return True


def ensure_nltk_resources(verbose: bool = True):
    # For METEOR and tokenization, these help avoid runtime errors
    resources = ["punkt", "wordnet", "omw-1.4"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}") if r == "punkt" else nltk.data.find(
                f"corpora/{r}"
            )
        except LookupError:
            if verbose:
                print(f"[INFO] downloading NLTK resource: {r}")
            nltk.download(r, quiet=True)


def process_submission_data(prompts: pathlib.Path, submission: pathlib.Path):
    data = []
    enc = tiktoken.get_encoding("o200k_harmony")
    with open(prompts, "r") as f_prompts, open(submission, "r") as f_subm:
        for prompt, token_ids in zip(f_prompts, f_subm):
            ids = [int(x) for x in token_ids.strip().split()]
            entry = {
                "id": hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest(),
                "prompt": prompt.strip(),
                "completion": enc.decode(ids),
            }
            data.append(entry)

    return data


def load_jsonl(path: pathlib.Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def coverage_report(subm: List[Dict], refs: List[Dict]):
    subm_ids = [d["id"] for d in subm]
    refs_ids = [d["id"] for d in refs]
    # keep only prompts from subm that also appear in ref
    filtered = [x for x in subm_ids if x in set(refs_ids)]
    coverage = len(filtered) / len(refs_ids) * 100.0
    print(
        f"[COVERAGE] Your submission covers {coverage:.1f}% of the prompts ({len(filtered)}/{len(refs_ids)})"
    )
    return filtered


def distinct_n(text: str, n: int) -> float:
    toks = text.split()
    if len(toks) < n:
        return 0.0
    ngrams = set(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return len(ngrams) / max(1, (len(toks) - n + 1))


def repetition_rate(text: str, min_run: int = 3) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    rep = 0
    run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
        else:
            if run >= min_run:
                rep += run
            run = 1
    if run >= min_run:
        rep += run
    return rep / len(toks)


def eval_reference_based(ids: List[str], refs: List[Dict], subm: List[Dict]):
    ensure_nltk_resources()
    meteor = evaluate.load("meteor")
    bscore = evaluate.load("bertscore")
    ids_set = set(ids)

    refs_dict = {d["id"]: d["completion"] for d in refs if d["id"] in ids_set}
    preds_dict = {d["id"]: d["completion"] for d in subm if d["id"] in ids_set}

    preds_completion = [preds_dict[i] for i in ids]
    gts_completion = [refs_dict[i] for i in ids]

    meteor_value = meteor.compute(
        predictions=preds_completion, references=gts_completion
    )["meteor"]
    bs = bscore.compute(
        predictions=preds_completion,
        references=gts_completion,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
    )
    d1 = [distinct_n(p, 1) for p in preds_completion]
    d2 = [distinct_n(p, 2) for p in preds_completion]
    rep = [repetition_rate(p) for p in preds_completion]
    length = [len(p.split()) for p in preds_completion]

    agg = {
        "items": len(ids),
        "METEOR": round(float(meteor_value), 6),
        "BERTScore_F1": round(statistics.fmean(bs["f1"]), 6),
        "BERTScore_P": round(statistics.fmean(bs["precision"]), 6),
        "BERTScore_R": round(statistics.fmean(bs["recall"]), 6),
        "distinct1": round(statistics.fmean(d1), 6),
        "distinct2": round(statistics.fmean(d2), 6),
        "repetition_rate": round(statistics.fmean(rep), 6),
        "avg_len_tokens": round(statistics.fmean(length), 3),
    }
    # Create DataFrame
    df = pd.DataFrame(list(agg.items()), columns=["Metric", "Value"])
    print(df)


def main():
    # parse CLI args
    args = parseCLI()
    # process submission file, which is lines of token ids
    subm = process_submission_data(args.prompts, args.submission)
    refs = load_jsonl(args.references)
    if not validate_references(refs, pathlib.Path("schema.json")):
        print("Exiting due to schema validation failure.")
        return
    # coverage report
    ids = coverage_report(subm, refs)
    eval_reference_based(ids, refs, subm)


if __name__ == "__main__":
    main()
