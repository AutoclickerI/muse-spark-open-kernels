"""Pack producer MMA and warpgroup variants into one synthetic diagnostic."""
from pathlib import Path
import ast
import runpy

root = Path(__file__).resolve().parents[1]
for builder in ('build_xqa_warp.py', 'build_xqa_group.py'):
    runpy.run_path(str(root / 'dev' / builder), run_name='__main__')
base = (root / 'dev/xqa_warp_candidate.py').read_text()
tree = ast.parse((root / 'dev/xqa_group_candidate.py').read_text())
gc = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == 'X_CUDA' for t in n.targets))
text = base[:base.index('\n_DECODE_FN = kernel_fn')] + '\n_DECODE_FN = kernel_fn\n'
text += 'G_CUDA = r"""\n' + gc + '\n"""\n'
text += (root / 'dev/xqa_next_tail.py').read_text()
start = text.index('"""'); end = text.index('"""', start + 3) + 3
text = text[:start] + '''"""Muse verify: asynchronous page-1 producers and swap-AB consumers.

All 64 logical queries occupy N, every cache key occupies M. The K128
mma.sync family has eight consumer warps; the K64 WGMMA family has one
consumer warpgroup. Both have 128 producer threads, cp.async full/empty
mbarriers, at least 32 KiB raw KV in flight, exact causal/window masks,
FP32 softmax and in-kernel last-CTA (max,sum,output) merge. QK uses raw
E4M3 keys and allowed E4M3 queries; PV widens V exactly to FP16. On sm_120
the WGMMA operations use matching mma.sync stand-ins in the same kernel.
prepare chooses per shape on synthetic scattered pages against FA3.
The incumbent q_len=1 path is preserved.
"""''' + text[end:]
(root / 'dev/xqa_next_candidate.py').write_text(text)
print('Built dev/xqa_next_candidate.py')
