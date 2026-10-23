"""Existing workload teacher-forcing harness, with bounded local GPU memory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving/harness"))
import teacher_forced_workload

launch = teacher_forced_workload.run.launch


def local_launch(args, *rest, **kwargs):
    args.mem_fraction_static = 0.3
    args.max_total_tokens = 100000
    return launch(args, *rest, **kwargs)


teacher_forced_workload.run.launch = local_launch
if __name__ == "__main__":
    teacher_forced_workload.main()
