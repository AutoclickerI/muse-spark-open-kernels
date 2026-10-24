"""Force receipt-backed expert prepare selection with real SM120 arithmetic.

The cold graphs/oracles here execute on SM120, not Hopper; selection is a glue
and correctness check only. No timing from this file is H200 evidence.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'problems/muse_prefill_layer'))
import reference as ref

candidate=ROOT/'problems/muse_prefill_layer/cand_mid_sweep.py'
spec=importlib.util.spec_from_file_location('mid_sweep',candidate)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def main():
    weights=ref.generate_weights(ref.WEIGHTS_SEED)
    paired=[];original_times=p._score_times
    def timing(graphs,refresh,flush,repeats=3,control=None):
        result=original_times(graphs,refresh,flush,repeats,control)
        if len(graphs)>1:
            assert repeats==20 and control is not None
            paired.append((len(graphs),repeats))
        return result
    original_prepare=p.prefill_gemm_prepare
    def untuned_dense(w,**kw):
        return original_prepare(w,tokens=kw['tokens'],budget_s=0,incumbent=kw['incumbent'])
    with patch.object(p,'IS_HOPPER',True),patch.object(p,'_score_times',side_effect=timing), \
         patch.object(p,'_tune_score_dense',lambda *a:None), \
         patch.object(p,'_tune_score_attention',lambda *a:None), \
         patch.object(p,'prefill_gemm_prepare',side_effect=untuned_dense), \
         patch.object(p,'_pg_tune_dense_groups',lambda *a:None):
        state=p.prepare(weights)
    pairs=copy.deepcopy(state['_expert_sweep_pairs'])
    assert len(pairs)==3,pairs
    for tokens,choices in p._SWEEP_EXPERT_CHOICES.items():
        assert set(choices[:2])<=set(pairs[tokens]['choices']),pairs[tokens]
        assert len(pairs[tokens]['fixtures'])==3 and pairs[tokens]['repeats']==20
    assert paired==[(3,20),(2,20),(3,20)],paired
    # Run whole-layer graphs using the freshly selected portable dispatch.
    ref._fa3_scheduler_metadata=lambda *a,**kw:None
    cases=[dict(batch=1,ctx=3000,layer='full',q_len=753,page=1),
           dict(batch=4,ctx=2500,layer='sliding',q_len=753,page=1),
           dict(batch=1,ctx=0,layer='full',q_len=16384,page=1)]
    checks=[]
    for case in cases:
        fields=ref.case_fields({**case,'seed':40721},weights)
        data=ref.generate_input(**case,**fields,seed=40721)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):p.kernel_fn(data,state)
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):output=p.kernel_fn(data,state)
        for seed in (40722,40723):
            fields=ref.case_fields({**case,'seed':seed},weights)
            fresh=ref.generate_input(**case,**fields,seed=seed)
            for key,value in fresh.items():
                if isinstance(value,torch.Tensor):data[key].copy_(value)
            graph.replay();torch.cuda.synchronize()
            ok,detail=ref.check_implementation(data,ref.collect(data,output),weights)
            assert ok,(case,detail)
            checks.append(dict(tokens=data['config']['tokens'],ok=ok,detail=detail))
    state['_deadline']=time.monotonic()+180
    state['_generator']=torch.Generator(device='cuda').manual_seed(478191)
    state['_measurements']={'expert':{}}
    flush=torch.zeros(1<<30,device='cuda',dtype=torch.uint8)
    original_expert=p._score_expert
    def failing(*args):
        if args[-1]=='tma_cluster':raise RuntimeError()
        return original_expert(*args)
    with patch.object(p,'_SWEEP_EXPERT_CHOICES',{753:p._SWEEP_EXPERT_CHOICES[753]}), \
         patch.object(p,'_score_expert',side_effect=failing):
        p._tune_score_experts(state,flush)
    assert 'tma_cluster' not in state['_expert_sweep_pairs'][753]['choices']
    with patch.object(p,'_SWEEP_EXPERT_CHOICES',{753:p._SWEEP_EXPERT_CHOICES[753]}), \
         patch.object(p,'_expert_realistic',side_effect=RuntimeError()):
        p._tune_score_experts(state,flush)
    assert state['_expert_sweep_pairs']=={}
    state['_deadline']=time.monotonic()-1
    p._tune_score_experts(state,flush)
    assert state['_expert_sweep_pairs']=={}
    report=dict(ok=True,sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        real_portable_paired_graphs=paired,fixtures_per_shape=3,oracle_sample_rows=64,
        selected_fresh_whole_layer_graphs=checks,launch_failure_rejected=True,
        budget_rejected=True,empty_message_error_rejected=True,graph_objects_active=p._SweepExpertGraph.active,
        local_dispatch={str(k):v['selected'] for k,v in pairs.items()},
        note='SM120 correctness/glue only; local times are not H200 ranking')
    assert p._SweepExpertGraph.active==0
    (ROOT/'runs/screen/mid_sweep_glue.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
