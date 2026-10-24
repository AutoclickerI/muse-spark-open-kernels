"""Run immutable source snapshots with all admitted serving Python ABIs forced."""
import sys
import re
import json
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'serving/tests'))
import attn_ar_harness as wrapper
import run

# Reuse the read-only harness and its logprob probe; replace only its kernel hook.
original_launch = wrapper.launch


def launch(args, tree, out, port):
    # GPU0 may spend the judge's normal fraction. TP2 shares GPU1 and must
    # stay at0.3. The300K token cap still bounds actual local KV allocation.
    args.mem_fraction_static = 0.85 if args.tp == 1 else 0.3
    if tree != run.common.STOCK_TREE:
        args.env = [v.replace('serving/tests/attn_ar_hook', 'serving/tests/integration_hook')
                    for v in args.env]
    return original_launch(args, tree, out, port)


original_ready = run.wait_ready


def ready(proc, port, log, timeout):
    elapsed = original_ready(proc, port, log, timeout)
    capacities = set(re.findall(r"max_total_num_tokens=(\d+)", Path(log).read_text()))
    resources = {"actual_kv_slots": sorted(capacities)}
    for name, query in (
        ("gpus", "--query-gpu=index,memory.used,memory.total,utilization.gpu"),
        ("processes", "--query-compute-apps=pid,gpu_uuid,used_memory"),
    ):
        resources[name] = subprocess.check_output(
            ["nvidia-smi", query, "--format=csv,noheader"], text=True
        ).splitlines()
    (Path(log).parent / "resources_start.json").write_text(json.dumps(resources, indent=2))
    if capacities != {"300000"}:
        raise RuntimeError(f"controlled workload requires300000 actual KV slots, got {capacities}")
    return elapsed


run.wait_ready = ready


wrapper.launch = launch
if __name__ == '__main__':
    if '--forced' in sys.argv:
        sys.argv.remove('--forced')
        import teacher_forced_workload
        teacher_forced_workload.main()
    else:
        run.main()
