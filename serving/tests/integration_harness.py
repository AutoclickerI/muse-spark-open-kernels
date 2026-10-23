"""Run immutable source snapshots with all admitted serving Python ABIs forced."""
import sys
import re
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
    if capacities != {"300000"}:
        raise RuntimeError(f"controlled workload requires300000 actual KV slots, got {capacities}")
    return elapsed


run.wait_ready = ready


wrapper.launch = launch
if __name__ == '__main__':
    run.main()
