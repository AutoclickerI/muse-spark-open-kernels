"""Build a standalone check; only the diagnostic differs from cand_v9_phase."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
base = (ROOT / 'problems/muse_moe_w4a16/cand_v9_phase.py').read_text()
serving = (ROOT / 'serving/sglang/python/sglang/srt/layers/muse_spark_moe.py').read_text()
adapter = (ROOT / 'serving/sglang/python/sglang/srt/layers/muse_spark_serving.py').read_text()
marlin = (ROOT / 'problems/muse_moe_w4a16/baseline.py').read_text()

def function(src, name):
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)

def renamed(src, mapping):
    return re.sub(r'\b(' + '|'.join(map(re.escape, mapping)) + r')\b', lambda m: mapping[m[0]], src)

# Independent, source-identical serving CUDA module and raw-dense ABI.
prefix = serving[:serving.index('GROUPED_CUDA_SRC =')]
prefix = renamed(prefix, {'CUDA_SRC':'_SERVING_CUDA_SRC', 'CPP_SRC':'_SERVING_CPP_SRC',
    '_module':'_serving_module', '_MOD':'_SERVING_MOD', '_repack':'_serving_repack',
    '_encode_scales':'_serving_encode_scales', '_bit_table':'_serving_bit_table',
    '_TABLE':'_SERVING_TABLE', '_DQ_SLOTS':'_SERVING_DQ_SLOTS', '_LA_MASK':'_SERVING_LA_MASK'})
raw = renamed(function(serving, '_v9_call'), {'_v9_call':'_serving_v9_call'})
# Keep the diagnostic's bounded V9 producer independent of concurrent changes
# to serving's outer expert portfolio. This is the accepted512-row adapter.
chunk = Path(__file__).with_name('serving_prefill_v9_adapter.py').read_text()
# Copy the stock math explicitly; no importing a repo baseline in the submission.
stock = marlin[marlin.index('from typing'):]
stock = renamed(stock, {'prepare':'_marlin_prepare', 'kernel_fn':'_marlin_forward'})
# Retain the scored kernel and all its CUDA sources byte-for-byte. Its old tracing
# diagnostic is replaced by this serving gate, with precisely the same dispatch.
tree = ast.parse(base)
lines = base.splitlines(keepends=True)
for node in sorted([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('prepare', '_diagnose')], key=lambda n:n.lineno, reverse=True):
    lines[node.lineno-1:node.end_lineno] = []
base = ''.join(lines)
tail = (Path(__file__).with_name('serving_prefill_diag_tail.py')).read_text()
out = base + '\n\n# Embedded serving source, independent of the image checkout.\n' + prefix + '\n' + raw + '\n\n' + chunk + '\n\n' + stock + '\n\n' + tail
path = ROOT / 'problems/muse_moe_w4a16/cand_serving_prefill_diag.py'
path.write_text(out)
assert function(out, 'kernel_fn') == function((ROOT / 'problems/muse_moe_w4a16/cand_v9_phase.py').read_text(), 'kernel_fn')
print(path, len(out.encode()), 'bytes; scored kernel_fn byte-identical')
