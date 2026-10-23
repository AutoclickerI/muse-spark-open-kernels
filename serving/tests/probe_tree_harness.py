"""Run the read-only lattice probe with normal CUDA and prefill graphs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving/harness"))
import run

launch = run.launch


def probe_launch(args, tree, output, port):
    args.mem_fraction_static = 0.3
    args.max_total_tokens = 300000
    args.env += [
        f"PYTHONPATH={tree / 'python'}:{ROOT / 'serving/tests/tree_probe_hook'}",
        f"MUSE_TREE_PROBE_DIR={output / 'lattices'}",
    ]
    return launch(args, tree, output, port)


run.launch = probe_launch
if __name__ == "__main__":
    run.main()
