"""Run native or exact-shape projection glue with stock prefill graphs enabled."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "serving/harness"))
import run

launch = run.launch
standin = "--standin" in sys.argv
if standin:
    sys.argv.remove("--standin")


def local_launch(args, tree, output, port):
    args.mem_fraction_static = 0.3
    args.max_total_tokens = 300000
    if args.tp > 1:
        # Documented harness setting: another project owns memory on GPU1.
        args.env += ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0"]
    if standin and tree == run.common.PATCH_TREE:
        args.env += [
            f"PYTHONPATH={tree / 'python'}:{ROOT / 'serving/tests/serving_projection_hook'}:{ROOT / 'serving/harness/pyhook'}"
        ]
    return launch(args, tree, output, port)


run.launch = local_launch
if __name__ == "__main__":
    if "--single-forced" in sys.argv:
        sys.argv.remove("--single-forced")
        import teacher_forced

        teacher_forced.main()
    elif "--forced" in sys.argv:
        sys.argv.remove("--forced")
        import teacher_forced_workload

        teacher_forced_workload.main()
    else:
        run.main()
