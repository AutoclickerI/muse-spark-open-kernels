"""Compile the changed Hopper projection/gather interfaces without executing them."""
from sglang.kernels.jit.utils import override_jit_cuda_arch
from sglang.kernels.ops.muse_spark_v12.gemm_ag import _jit_module
from sglang.kernels.ops.muse_spark_v12.projection_hopper import _module

with override_jit_cuda_arch(9, 0, suffix='a'):
    _module('9.0a')
    for n in (4096, 8192):
        _jit_module(n, 8)
        print('sm_90a multicast gather compiled', n, flush=True)
print('sm_90a router compiled', flush=True)

# These run after compilation, before the timed server bracket.
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('local_eval', root / 'tools/local_eval.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)
inner = root / 'serving/sglang'
paths = subprocess.check_output(['git', '-C', str(inner), 'diff', '--name-only',
                                 'a55ab7d5', '--', 'python/sglang'], text=True).splitlines()
results = {}
for name in paths:
    file = inner / name
    if file.suffix == '.py' and file.exists():
        results[name] = checks.undefined_names(file) + checks.call_signatures(file)
        results[name] += checks.hopper_compile(file)
out = Path('/data2/jihoon/muse_kernel/harness/results/upstream-r01-evidence')
(out / 'static_checks.json').write_text(json.dumps(results, indent=2)+'\n')
assert all(row['ok'] for group in results.values() for row in group), results
print('F821-F823, call signatures and all embedded CUDA sm90a: PASS', flush=True)
for test in ('test_upstream_serving_policy.py', 'test_hopper_glue.py'):
    subprocess.run([sys.executable, str(root / 'serving/tests' / test)], check=True)
