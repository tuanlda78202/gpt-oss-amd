#!/usr/bin/env python3
"""
test_tokenizer.py — Compare C (test_tokenizer) vs Python (tiktoken) encode/decode.

Usage:
  python test_tokenizer.py --bin ./test_tokenizer --tok ./tokenizer.bin \
      --prompts prompts.txt --encoding o200k_harmony --verbose

Exit codes:
  0 = all encode/decode pairs match
  1 = at least one mismatch
  2 = usage / environment error (missing files, bad args)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable, List, Tuple

import tiktoken

# ----------------------------
# Data structures
# ----------------------------


@dataclass(frozen=True)
class CaseResult:
    prompt: str
    c_ids: List[int]
    py_ids: List[int]
    c_text: str
    py_text: str

    @property
    def encode_match(self) -> bool:
        return self.c_ids == self.py_ids

    @property
    def decode_match(self) -> bool:
        return self.c_text == self.py_text


# ----------------------------
# Helpers
# ----------------------------


def require_file(path: Path, kind: str) -> None:
    if not path.is_file():
        print(f"[ERROR] {kind} not found: {path}", file=sys.stderr)
        sys.exit(2)


def run_cmd(cmd: List[str], timeout: float) -> str:
    try:
        out = subprocess.check_output(
            cmd, encoding="utf-8", errors="ignore", timeout=timeout
        )
        return out.strip()
    except FileNotFoundError:
        print(f"[ERROR] Executable not found: {cmd[0]}", file=sys.stderr)
        sys.exit(2)
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command timed out: {shlex.join(cmd)}", file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Command failed ({e.returncode}): {shlex.join(cmd)}",
            file=sys.stderr,
        )
        if e.output:
            print(e.output, file=sys.stderr)
        sys.exit(2)


def run_c_encoder(binary: Path, tokbin: Path, text: str, timeout: float) -> List[int]:
    cmd = [str(binary), "-t", str(tokbin), "-i", text]
    out = run_cmd(cmd, timeout)
    return [int(x) for x in out.split()] if out else []


def run_c_decoder(binary: Path, tokbin: Path, text: str, timeout: float) -> str:
    cmd = [str(binary), "-t", str(tokbin), "-i", text, "-r"]
    out = run_cmd(cmd, timeout)
    lines = out.splitlines()
    return lines[-1] if lines else ""


def first_diff(a: List[int], b: List[int]) -> Tuple[int, int | None, int | None]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[i], b[i]
    if len(a) != len(b):
        return n, (a[n] if n < len(a) else None), (b[n] if n < len(b) else None)
    return -1, None, None


def iter_prompts(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


# ----------------------------
# Core compare
# ----------------------------


def compare_one(
    binary: Path, tokbin: Path, enc: tiktoken.Encoding, prompt: str, timeout: float
) -> CaseResult:
    c_ids = run_c_encoder(binary, tokbin, prompt, timeout)
    py_ids = enc.encode_ordinary(prompt)
    c_decoded = run_c_decoder(binary, tokbin, prompt, timeout)
    py_decoded = enc.decode(py_ids)
    return CaseResult(prompt, c_ids, py_ids, c_decoded, py_decoded)


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare C tokenizer vs tiktoken for encode/decode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bin", required=True, help="Path to C test_tokenizer binary")
    p.add_argument("--tok", required=True, help="Path to tokenizer.bin")
    p.add_argument(
        "--prompts",
        default="prompts.txt",
        help="Path to prompts file (one prompt per line)",
    )
    p.add_argument(
        "--encoding", default="o200k_harmony", help="tiktoken encoding to use"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Only test the first N prompts"
    )
    p.add_argument("--verbose", action="store_true", help="Print all details")
    p.add_argument(
        "--only-mismatches",
        action="store_true",
        help="Print details only for mismatches",
    )
    p.add_argument(
        "--timeout", type=float, default=10.0, help="Per-command timeout in seconds"
    )
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    args = parse_args()

    current_dir = Path(__file__).resolve().parent
    bin_path = current_dir / args.bin
    tok_path = current_dir / args.tok
    prm_path = current_dir / args.prompts

    require_file(bin_path, "binary")
    require_file(tok_path, "tokenizer")
    require_file(prm_path, "prompts file")

    try:
        enc = tiktoken.get_encoding(args.encoding)
    except Exception as e:
        print(
            f"[ERROR] Unknown tiktoken encoding '{args.encoding}': {e}", file=sys.stderr
        )
        sys.exit(2)

    # Sanity: enc.name can differ subtly across versions; trust user-provided name.

    prompts = list(iter_prompts(prm_path))
    if args.limit is not None:
        prompts = prompts[: max(args.limit, 0)]

    enc_miss = 0
    dec_miss = 0

    for prompt in prompts:
        res = compare_one(bin_path, tok_path, enc, prompt, args.timeout)

        is_mismatch = (not res.encode_match) or (not res.decode_match)
        should_print = (
            args.verbose
            or (args.only_mismatches and is_mismatch)
            or (not args.only_mismatches)
        )

        if should_print:
            print(f"PROMPT: {prompt!r}")
            print(f"  C  encoded: {res.c_ids}")
            print(f"  PY encoded: {res.py_ids}")
            print(f"  C  decoded: {res.c_text!r}")
            print(f"  PY decoded: {res.py_text!r}")

        if res.encode_match:
            if should_print:
                print("  [ENCODE MATCH]")
        else:
            enc_miss += 1
            idx, av, bv = first_diff(res.c_ids, res.py_ids)
            if should_print:
                print("  [ENCODE MISMATCH]")
                if idx >= 0:
                    print(f"    First diff at idx {idx}: C={av} PY={bv}")

        if res.decode_match:
            if should_print:
                print("  [DECODE MATCH]")
        else:
            dec_miss += 1
            if should_print:
                print("  [DECODE MISMATCH]")

        if should_print:
            print("-" * 60)

    total = len(prompts)
    print(
        f"\nSummary: "
        f"{total - enc_miss}/{total} encode matched, "
        f"{total - dec_miss}/{total} decode matched."
    )
    sys.exit(0 if (enc_miss == 0 and dec_miss == 0) else 1)


if __name__ == "__main__":
    main()
