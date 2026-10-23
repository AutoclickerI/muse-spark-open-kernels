"""Paths and constants shared by the serving harness scripts.

The harness keeps its bulky outputs (tiny checkpoints, server logs, results) outside the repo,
on /data2, and runs everything inside the judge's image through tools/img.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().parent
PATCH_TREE = REPO / "serving" / "sglang"          # our patch (the working tree we edit)
BASE_COMMIT = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"

DATA = Path(os.environ.get("MUSE_HARNESS_DATA", "/data2/jihoon/muse_kernel/harness"))
STOCK_TREE = Path(os.environ.get("MUSE_STOCK_TREE", "/data2/jihoon/muse_kernel/sglang-stock"))
MODELS = DATA / "models"
RESULTS = DATA / "results"

# Weight sets: "random" = seeded random weights (as asked); "chain" = structured weights whose
# greedy output has wide margins and that the drafter mostly predicts (see make_tiny.py).
WEIGHT_SETS = {
    "random": ("muse-tiny-nvfp4", "dflash2-tiny"),
    "chain": ("muse-tiny-nvfp4-chain", "dflash2-tiny-chain"),
}


def model_dirs(weights: str) -> tuple[Path, Path]:
    target, draft = WEIGHT_SETS[weights]
    return MODELS / target, MODELS / draft

# Tokens the tiny tokenizer reserves (ids follow the real Muse config / DFlash2 config).
BOS_ID = 200000
EOS_ID = 200001
MASK_ID = 200002
HEADER_START_ID = 200005
HEADER_END_ID = 200006
EOT_ID = 200008
PAD_ID = 200018
VOCAB = 202048
NUM_WORD_TOKENS = 200000                           # ids [0, 200000) are plain "w<id>" words

OUTPUT_TOKENS = 220                                # the judge's per-request output length


def tree_for(name: str) -> Path:
    """'stock' -> the pristine a55ab7d5 worktree; 'patch' -> serving/sglang; else a path."""
    if name == "stock":
        return STOCK_TREE
    if name == "patch":
        return PATCH_TREE
    return Path(name).resolve()


def checkpoint_fingerprint(directory: Path) -> dict:
    """Content identity, including the actual weight bytes, for reproducible comparisons."""
    files = {}
    for path in sorted(directory.iterdir()):
        if path.suffix not in ('.json', '.safetensors', '.jinja'):
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        files[path.name] = digest.hexdigest()
    return files
