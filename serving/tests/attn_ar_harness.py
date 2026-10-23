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
stop = run.stop
with_logprob = "--with-logprob" in sys.argv
if with_logprob:
    sys.argv.remove("--with-logprob")


def local_launch(args, tree, output, port):
    args.mem_fraction_static = 0.3
    args.max_total_tokens = 300000
    if args.tp > 1:
        args.env += ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0"]
    if tree != run.common.STOCK_TREE:
        args.env += [
            f"PYTHONPATH={tree / 'python'}:{ROOT / 'serving/tests/attn_ar_hook'}:{ROOT / 'serving/harness/pyhook'}"
        ]
    proc, cmd = launch(args, tree, output, port)
    proc.logprob_probe = (args, tree, output, port)
    return proc, cmd


def local_stop(proc):
    try:
        if (
            with_logprob
            and getattr(proc, "harness_fired", False)
            and proc.poll() is None
        ):
            from harness_logprob_probe import collect

            collect(run, *proc.logprob_probe)
    finally:
        stop(proc)


run.launch = local_launch
run.stop = local_stop
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
