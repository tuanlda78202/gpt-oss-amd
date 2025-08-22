"""Export a tiktoken vocabulary to a compact binary format.

Binary layout (little-endian):
    int32  max_token_length
    repeat n_vocab times:
        float32 score
        int32   byte_len
        bytes   token_bytes

Conventions:
- Normal tokens: score = merge rank (fallback to token id if rank missing).
- Special tokens: score = SPECIAL_TOKEN_SCORE and bytes = UTF-8 text.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tiktoken

# ----------------------------
# Constants & simple types
# ----------------------------

SPECIAL_TOKEN_SCORE: float = -1e30
I32_LE: str = "<i"
F32_LE: str = "<f"


@dataclass(frozen=True)
class TokenRecord:
    score: float
    data: bytes


# ----------------------------
# Core logic
# ----------------------------


def build_id_maps(
    enc: "tiktoken.core.Encoding",
) -> Tuple[List[Optional[bytes]], List[Optional[str]]]:
    """Return parallel maps: id -> raw bytes, id -> special text (if any)."""
    n_vocab = enc.n_vocab
    id_to_bytes: List[Optional[bytes]] = [None] * n_vocab
    id_to_text: List[Optional[str]] = [None] * n_vocab

    # Private fields are stable in practice across tiktoken versions.
    merge_ranks: Dict[bytes, int] = getattr(enc, "_mergeable_ranks", {}) or {}
    special_tokens: Dict[str, int] = getattr(enc, "_special_tokens", {}) or {}

    # Normal tokens by merge table
    for token_bytes, tid in merge_ranks.items():
        if 0 <= tid < n_vocab:
            id_to_bytes[tid] = token_bytes

    # Special tokens by id
    for text, tid in special_tokens.items():
        if 0 <= tid < n_vocab:
            id_to_text[tid] = text

    # Fill remaining holes defensively
    for tid in range(n_vocab):
        if id_to_bytes[tid] is None and id_to_text[tid] is None:
            try:
                id_to_bytes[tid] = enc.decode_single_token_bytes(tid)
            except Exception:
                # Last resort: scan specials by id
                for text, sid in special_tokens.items():
                    if sid == tid:
                        id_to_text[tid] = text
                        break

    return id_to_bytes, id_to_text


def build_token_records(
    enc: "tiktoken.core.Encoding",
    id_to_bytes: List[Optional[bytes]],
    id_to_text: List[Optional[str]],
) -> Tuple[List[TokenRecord], int]:
    """Materialize TokenRecord list and compute max token byte length."""
    n_vocab = enc.n_vocab
    merge_ranks: Dict[bytes, int] = getattr(enc, "_mergeable_ranks", {}) or {}

    records: List[TokenRecord] = []
    max_len = 0

    for tid in range(n_vocab):
        special_text = id_to_text[tid]
        if special_text is not None:
            # Special tokens: store UTF-8 *text* so the C side can match them by name
            data = special_text.encode("utf-8", errors="strict")
            score = SPECIAL_TOKEN_SCORE
        else:
            token_bytes = id_to_bytes[tid]
            if token_bytes is None:
                raise RuntimeError(
                    f"Token id {tid} unresolved: missing bytes and special text."
                )

            # === IMPORTANT: store RAW BYTES for normal tokens ===
            # This makes the C BPE concatenate bytes and hit the exact entries.
            # Only special-case the true NUL byte (rare / unused in UTF-8 text),
            # which cannot live inside C strings safely.
            if len(token_bytes) == 1 and token_bytes[0] == 0x00:
                data = b"<0x00>"  # harmless stand-in; won't affect UTF-8 inputs
            else:
                data = token_bytes

            # Score encodes merge priority (lower rank = earlier merge).
            # Keep as the rank float; the C side will interpret consistently.
            score = float(merge_ranks.get(token_bytes, tid))

        records.append(TokenRecord(score=score, data=data))
        if len(data) > max_len:
            max_len = len(data)

    return records, max_len


def write_tokenizer_binary(
    out_path: Path, records: List[TokenRecord], max_token_length: int
) -> None:
    """Write the tokenizer.bin file atomically."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        f.write(struct.pack(I32_LE, max_token_length))
        for rec in records:
            f.write(struct.pack(F32_LE, rec.score))
            f.write(struct.pack(I32_LE, len(rec.data)))
            f.write(rec.data)
    tmp_path.replace(out_path)


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tiktoken vocab to tokenizer.bin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--out", required=True, help="Output tokenizer.bin path")
    parser.add_argument("--encoding", default="o200k_harmony", help="tiktoken encoding")
    parser.add_argument(
        "--check-vocab",
        type=int,
        default=None,
        help="If set, assert that n_vocab equals this value.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help=(
            "Override max_token_length. Must be >= longest token; "
            "otherwise the program exits with error."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        enc = tiktoken.get_encoding(args.encoding)
        for name in [
            "<|start|>",
            "<|end|>",
            "<|return|>",
            "<|message|>",
            "<|channel|>",
            "<|constrain|>",
            "<|endoftext|>",
        ]:
            tid = enc._special_tokens.get(name)
            if tid is not None:
                print(f"{name}: {tid}")
    except Exception as e:
        print(f"[ERROR] Unknown encoding '{args.encoding}': {e}", file=sys.stderr)
        sys.exit(2)

    if args.check_vocab is not None and enc.n_vocab != args.check_vocab:
        print(
            f"[ERROR] n_vocab mismatch: got {enc.n_vocab}, expected {args.check_vocab}",
            file=sys.stderr,
        )
        sys.exit(3)

    id_to_bytes, id_to_text = build_id_maps(enc)
    records, computed_max_len = build_token_records(enc, id_to_bytes, id_to_text)

    max_token_length = computed_max_len if args.max_len is None else args.max_len
    if max_token_length < computed_max_len:
        print(
            f"[ERROR] --max-len={max_token_length} < required={computed_max_len} "
            f"(longest token byte length).",
            file=sys.stderr,
        )
        sys.exit(4)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_tokenizer_binary(out_path, records, max_token_length)

    print(
        f"[OK] Wrote tokenizer with {len(records)} tokens to {out_path} "
        f"(max_token_length={max_token_length})"
    )


if __name__ == "__main__":
    main()
