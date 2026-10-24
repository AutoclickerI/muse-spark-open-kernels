"""Fold the landed expert sweep into a fresh mid_dense candidate; leave sends frozen."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
base=ROOT/'problems/muse_prefill_layer/cand_mid_dense.py'
s=base.read_text()
tree=ast.parse(s)
node=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_tune_score_experts'][-1]
lines=s.splitlines(keepends=True)
lines[node.lineno-1:node.end_lineno]=['\n']
s=''.join(lines)
s+='\n'+(HERE/'expert_tuner.py.inc').read_text()+'\n'
s=s.replace("st['expert_dispatch']={753:'humming_masked',3012:'humming_masked',16384:'dg_hint_w8'}",
            "st['expert_dispatch']=dict(_SWEEP_EXPERT_DEFAULTS)",1)
needle="def _score_expert(pooled,probs,slot_e,slot_pos,counts,st,name):\n"
branch="""    if name=='tma_cluster':
        # This lab label means raw17 with4x the persistent launch grid.
        # It is not hardware cluster multicast or a different dequant format.
        raw=dict(st['raw_states']['raw17']);raw['sms']=int(raw['sms'])*4
        return _raw_experts_tiled(pooled,probs,slot_e,slot_pos,counts,raw)
"""
assert needle in s
s=s.replace(needle,needle+branch,1)
s=s.replace("    print('SCORE_LAST_SUMMARY',", "    _expert_sweep_last(st)\n    print('SCORE_LAST_SUMMARY',",1)
s=s.replace("str(error).splitlines()[0][:240],flush=True)",
            "(str(error).splitlines() or [type(error).__name__])[0][:240],flush=True)")
s=s.replace('Mid-experts2 portfolio with gated 16K pooler/unpooler and dense/router scheduling.',
            'Mid-dense portfolio with receipt-backed expert defaults and paired realistic-route selection.')
out=ROOT/'problems/muse_prefill_layer/cand_mid_sweep.py'
out.write_text(s)
manifest=dict(base=str(base.relative_to(ROOT)),base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
              candidate=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
              sweep_job='b964cb5d180a463b',cutlass_queue_key='63aa77388930a37e',cutlass_receipt=None)
(HERE/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
