"""Reach the actual Hopper-only CUTLASS hook in full layer prepare on sm120."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'problems/muse_prefill_layer'))
import reference

path=ROOT/'problems/muse_prefill_layer/cand_mid_cutlass.py'
spec=importlib.util.spec_from_file_location('cutlass_prepare',path)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
weights=reference.generate_weights(reference.WEIGHTS_SEED)
calls=[];pfc_original=p.pfc_prepare;pg_original=p.prefill_gemm_prepare

def pfc(st,deadline):
    assert deadline<=st['_deadline']
    assert len(p._EXPERT_CHOICES)==34
    calls.append('receipt_gated_cutlass')
    pfc_original(st,deadline)
    assert not st['_pfc_plans'] and 'no H200-positive' in st['_pfc_notes'][0]

def pg(weights,**kwargs):
    assert calls==['receipt_gated_cutlass']
    assert kwargs['tokens']==(16384,)
    calls.append('dense_and_group_portfolio')
    return pg_original(weights,tokens=kwargs['tokens'],budget_s=0,incumbent=kwargs['incumbent'])

with patch.object(p,'IS_HOPPER',True),patch.object(p,'_tune_score_dense',lambda *a:None), \
     patch.object(p,'_tune_score_experts',lambda *a:None),patch.object(p,'_tune_score_attention',lambda *a:None), \
     patch.object(p,'_pg_tune_dense_groups',lambda *a:None), \
     patch.object(p,'pfc_prepare',side_effect=pfc),patch.object(p,'prefill_gemm_prepare',side_effect=pg), \
     patch.object(p,'vgc_module',side_effect=AssertionError('an empty shortlist must not compile')):
    state=p.prepare(weights)
assert calls==['receipt_gated_cutlass','dense_and_group_portfolio'] and state['pg'] is not None
assert not state['_pfc_plans']
report=dict(ok=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hook_order=calls,
            shortlist='empty; incumbent retained; no CUTLASS runtime compile')
(ROOT/'runs/screen/prefill_cutlass_prepare.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
