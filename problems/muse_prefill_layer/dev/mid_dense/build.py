"""Merge exact mid_experts2 base with the frozen a30 dense integration."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
base=ROOT/'problems/muse_prefill_layer/cand_mid_experts2.py'
dense=ROOT/'problems/muse_prefill_layer/cand_dense_router_a30.py'
out=ROOT/'problems/muse_prefill_layer/cand_mid_dense.py'
s=base.read_text()
d=dense.read_text()

def function(src,name):
    node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '\n'.join(src.splitlines()[node.lineno-1:node.end_lineno])

s=s.replace('"""Score portfolio: multiple lossless NVFP4 layouts and per-shape cold-graph stage selection."""',
            '"""Mid-experts2 portfolio with gated 16K pooler/unpooler and dense/router scheduling."""')
start=s.index('import math\n')
stop=s.index('HIDDEN = ',start)
s=s[:start]+d[d.index('import ctypes as ct\n'):d.index('HIDDEN = ')]+s[stop:]
s=s.replace(function(s,'kernel_fn'), function(d,'kernel_fn'),1)
prepare=function(d,'prepare')
# Preserve both optional exact-route Humming plan banks from the new board best.
mid_start=s.index('    # Mid-range expert choice:')
mid_stop=s.index('    marlin_weights=',mid_start)
mid=s[mid_start:mid_stop]
body_start=mid.index('    mid=copy.copy(hum)')
body_stop=mid.index('    # The H200 profile',body_start)
body=mid[body_start:body_stop]
mid=mid[:body_start]+"    mid=None\n    try:\n"+'\n'.join('    '+line for line in body.rstrip().splitlines())+"\n    except Exception as error:\n        mid=None\n        print('SCORE_REJECT build humming_mid',type(error).__name__,str(error).splitlines()[0][:240],flush=True)\n"+mid[body_stop:]
mid=mid.replace('        mid_tma=copy.copy(mid);mid_tma.tma=True',
                "        if mid is None: raise RuntimeError('humming_mid build rejected')\n        mid_tma=copy.copy(mid);mid_tma.tma=True")
prepare=prepare.replace('    marlin_weights=',mid+'    marlin_weights=',1)
prepare=prepare.replace("16384:'dg_w8'", "16384:'dg_hint_w8'")
s=s.replace(function(s,'prepare'),prepare,1)
# Ruff detects deleted closure cells even when a branch is locally dormant.
s='\n'.join(line for line in s.splitlines() if not line.strip().startswith('del '))+'\n'
append=d[d.index('"""Prepared Muse TP8 prefill GEMMs;'):]
# The original sequence is inherited from mid_experts2, which shares a30's
# routing implementation. Use an unambiguous name in this candidate's receipts.
append=append.replace("'a30'", "'mid'").replace('a30_us','incumbent_us').replace("a30's","the incumbent's")
s += '\n'+append
out.write_text(s)
print(out.relative_to(ROOT))
