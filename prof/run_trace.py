"""Run the unmodified harness with Nsight injected into its server child."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving/harness"))
import run

original = run.subprocess.Popen


def popen(cmd, *args, **kwargs):
    if isinstance(cmd, list) and "sglang.launch_server" in cmd and os.getenv("MUSE_NSYS_OUTPUT"):
        at = cmd.index("python")
        cmd = cmd[:at] + ["nsys", "profile", "--trace=cuda,nvtx",
                          "--cuda-graph-trace=" + os.getenv("MUSE_NSYS_GRAPH_MODE", "graph"), "--sample=none",
                          "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
                          "--capture-range-end=stop", "--force-overwrite=true",
                          "--output=" + os.environ["MUSE_NSYS_OUTPUT"]] + cmd[at:]
    return original(cmd, *args, **kwargs)


run.subprocess.Popen = popen
if __name__ == "__main__":
    run.main()
