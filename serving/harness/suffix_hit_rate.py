#!/usr/bin/env python3
"""Estimate longest-suffix drafting opportunities on a repeated agentic trace.

The tiny Muse checkpoint ships a synthetic ``wN`` vocabulary.  We build a
multi-turn tool/code trace from those token strings and re-tokenize it with the
checkpoint tokenizer, so this measures the exact suffix algorithm and its
bounded 256-token history window without consulting model logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer


BLOCK = 8
OUTPUT = 220
TURNS = 13


def words(start: int, n: int) -> list[str]:
    return [f"w{(start + i) % 190000}" for i in range(n)]


def retokenize(parts: list[str], tokenizer: Tokenizer) -> list[int]:
    text = " ".join(parts)
    return tokenizer.encode(text).ids


def trace(tokenizer: Tokenizer) -> list[tuple[list[int], list[int]]]:
    """Return (prompt, 220-token continuation) pairs for thirteen turns."""
    tool = words(1000, 34)
    file_block = words(5000, 48)
    test_block = words(9000, 30)
    marker = words(70000, 3)
    out = []
    history: list[str] = []
    for turn in range(TURNS):
        # Each turn repeats a tool result, a file excerpt, and a test command;
        # the remaining words are a new request and a new assistant tail.
        user = words(20000 + turn * 211, 46)
        # The marker also closes the previous assistant trace.  It is within
        # the worker's 256-token window, so the next turn can copy the tool
        # result that followed the previous marker.
        prompt_parts = history + ["w200", *user, *marker]
        copied = tool + file_block + test_block
        novel = words(30000 + turn * 307, OUTPUT - len(copied) - len(marker) - 5)
        continuation_parts = copied + novel + marker + copied[:5]
        prompt = retokenize(prompt_parts, tokenizer)
        continuation = retokenize(continuation_parts, tokenizer)[:OUTPUT]
        if len(continuation) != OUTPUT:
            raise RuntimeError(f"continuation tokenized to {len(continuation)} tokens")
        out.append((prompt, continuation))
        history += prompt_parts + continuation_parts
    return out


def candidate(context: list[int], width: int = BLOCK) -> list[int]:
    """Match the worker's longest 2..8 suffix in its last 256 tokens."""
    context = context[-256:]
    for n in range(min(8, len(context) - 1), 1, -1):
        suffix = context[-n:]
        for pos in range(len(context) - n - 1, -1, -1):
            if context[pos : pos + n] == suffix:
                return context[pos + n : pos + n + width]
    return []


def measure(tokenizer_path: str) -> dict:
    tokenizer = Tokenizer.from_file(tokenizer_path)
    rows = trace(tokenizer)
    hits = [0] * BLOCK
    eligible = [0] * BLOCK
    lengths: list[int] = []
    blocks = 0
    for prompt, continuation in rows:
        for start in range(0, OUTPUT, BLOCK):
            pred = candidate(prompt + continuation[:start])
            blocks += 1
            if pred:
                eligible[0] += 1
            length = 0
            for i, token in enumerate(pred[: min(BLOCK, OUTPUT - start)]):
                if token != continuation[start + i]:
                    break
                length += 1
            lengths.append(length)
            for k in range(1, BLOCK + 1):
                if len(pred) >= k:
                    eligible[k - 1] += 1
                if length >= k:
                    hits[k - 1] += 1
    return {
        "turns": TURNS,
        "output_tokens_per_turn": OUTPUT,
        "verify_blocks": blocks,
        "coverage": [x / blocks for x in eligible],
        "hit_rate": [x / blocks for x in hits],
        "conditional_hit_rate": [
            hits[i] / eligible[i] if eligible[i] else 0.0 for i in range(BLOCK)
        ],
        "expected_match_tokens": sum(lengths) / blocks,
        "expected_tail_after_dflash_4_16": sum(
            max(0.0, length - 4.16) for length in lengths
        )
        / blocks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer",
        default="/data2/jihoon/muse_kernel/harness/models/muse-tiny-nvfp4/tokenizer.json",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    result = measure(args.tokenizer)
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
