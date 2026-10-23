"""Harness launcher exercising the SM90 AR ownership branch using NCCL+norm.

The stand-in changes only the kernel seam, forcing output-reduction ownership
through the new model branch on TP2. Keeps all stock prefill graph shapes enabled,
and a 0.3 memory fraction for the shared local GPU1.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving/harness"))
import run

launch = run.launch


def local_launch(args, tree, output, port):
    args.mem_fraction_static = 0.3
    args.max_total_tokens = 100000
    if not hasattr(args, "tree"):
        tree_name = "patch" if tree == run.common.PATCH_TREE else "stock"
    else:
        tree_name = args.tree
    if tree_name == "patch":
        args.env += [
            f"PYTHONPATH={tree / 'python'}:{ROOT / 'serving/tests/attn_ar_hook'}:{ROOT / 'serving/harness/pyhook'}"
        ]
    return launch(args, tree, output, port)


run.launch = local_launch
if __name__ == "__main__":
    if "--forced" in sys.argv:
        sys.argv.remove("--forced")
        import teacher_forced_workload

        teacher_forced_workload.main()
    else:
        run.main()
