#!/usr/bin/env python3
"""Build a decode-layer candidate from the measured exact-router lab winners. No remote send."""
import argparse
import hashlib
import json
from pathlib import Path
from build_moe_real_routing import ROOT,namespace,segment
from read_moe_real_routing import select_dispatch

CHOICES=tuple((w,s,f,n) for w,s in ((32,3),(64,3),(64,5),(32,5)) for f,n in ((1,8),(1,16),(.5,8)))


def build(records):
    rec,core,plan,measured=select_dispatch(records)
    names=set(plan.values());parts=[];p='problems/muse_moe_w4a16/'
    if 'v9p' in names:parts.append(namespace(segment(p+'cand_v9_phase.py',end='\nimport json'),'ph_'))
    if 'v19' in names:parts.append(namespace(segment(p+'dev/parked/cand_v19.py',end='\n# V9 is the native')+'\nTRACE_STRIDE = 10\n','v19_'))
    if 'v11' in names:
        s=segment(p+'cand_v11.py',start='\nimport json',end='\ndef _time_variant').replace('_prepare_native = _prepare_v9','').replace('_kernel_native = _kernel_v9','')
        parts.append(namespace(s,'hu_'))
    if 'humming' in names:parts.append(namespace(segment(p+'cand_humming.py'),'ho_'))
    if 'h7' in names:parts.append(namespace(segment(p+'cand_humming7.py'),'h7_'))
    if 'marlin' in names:parts.append(namespace(segment(p+'baseline.py'),'ma_'))
    if 'grouped' in names:
        s=segment('problems/muse_decode_layer/cand_r3_v9_cost_autotune.py',start='\nimport os\nimport triton\nimport triton.language as tl\n\nGROUPED_SRC',end='\ndef _time_variant')
        parts.append(namespace(s,'gr_'))
    if any(n.startswith('tg') for n in names):
        s=segment(p+'dev/parked/cand_tma_grouped.py',start='\nimport triton\nimport triton.language as tl',end='\ndef _tg_router_weights')
        a=s.index('\ndef _rms(');b=s.index('TG_CUDA =');s=s[:a]+'\n'+s[b:]
        s=s.replace('_tg_incumbent_prepare = prepare','').replace('_tg_incumbent_kernel = kernel_fn','')
        parts.append(namespace(s,'tg_'))
    setup=[];calls=[]
    for n in sorted(names-{'v9'}):
        if n=='grouped':
            setup.append("    state['lab_grouped']=dict(state['moe_bank'],grouped_mod=gr__grouped_module())")
            call="gr__grouped_call(data,state['lab_grouped'])"
        elif n=='v9p':
            setup.append("    state['lab_v9p']=dict(state['moe_bank'],mod=ph__module('native'))")
            call="ph__kernel_v9(data,state['lab_v9p'])"
        elif n=='v19':
            setup.append("    state['lab_v19']=dict(state['moe_bank'],mod=v19__module('native'))")
            call="v19__kernel_v9(data,state['lab_v19'])"
        elif n=='v11':
            setup.append("    state['lab_v11']=hu__prepare_humming(w)")
            call="hu__kernel_humming(data,state['lab_v11'])"
        elif n=='humming':
            setup.append("    state['lab_humming']=ho_prepare(w)")
            call="ho_kernel_fn(data,state['lab_humming'])"
        elif n=='h7':
            setup.append("    state['lab_h7']=h7_prepare(w)")
            call="h7_kernel_fn(data,state['lab_h7'])"
        elif n=='marlin':
            setup.append("    state['lab_marlin']=ma_prepare(w)")
            call="ma_kernel_fn(data,state['lab_marlin'])"
        elif n.startswith('tg'):
            call=f"tg__tg_run(data,state['lab_tg'],{CHOICES[int(n[2:])]!r})"
        else:raise ValueError(n)
        calls.append(f"    if name=={n!r}:return {call}")
    if any(n.startswith('tg') for n in names):
        setup.append("    state['lab_tg']=tg__tg_prepare_bank(w)")
        selected=tuple(CHOICES[int(n[2:])] for n in sorted(names) if n.startswith('tg'))
        setup.append(f"    for rows in range(1,513):\n        for choice in {selected!r}:tg__tg_configure(state['lab_tg'],rows,choice)")
    tail=f'''
# H200 labs loaded={rec['job']}, captured-core={core['job']}. Shape-only dispatch; canonical NVFP4 stays packed.
_LAB_MOE_PLAN = {plan!r}
_lab_original_prepare = prepare
_lab_original_experts = experts_v9


def prepare(weights):
    state = _lab_original_prepare(weights)
    w = dict(weights, post_norm_w=weights['post_expert_norm'])
{chr(10).join(setup)}
    print('MOE_LAB_DISPATCH',_LAB_MOE_PLAN,flush=True)
    return state


def experts_v9(x,wts,lids,state):
    m = int(x.shape[0])
    key = min(_LAB_MOE_PLAN,key=lambda t:abs(t-m))
    name = _LAB_MOE_PLAN[key]
    if name == 'v9':
        return _lab_original_experts(x,wts,lids,state)
    # Global expert APIs require a valid remote ID, rather than negative local IDs.
    ids = torch.where(lids < 0,255,lids).to(torch.int32)
    data = dict(hidden_states=x,topk_ids=ids,topk_weights=wts,
                config=dict(bs=m,local_count=32,d_hidden=4096,d_expert=4096,local_start=0,n_experts=256,top_k=8,sharding='ep8'))
{chr(10).join(calls)}
    raise ValueError(name)
'''
    base=ROOT/'problems/muse_decode_layer/cand_r3_v9.py'
    body=base.read_text()+'\n\n'+'\n'.join(parts)+'\n'+tail
    target=ROOT/'problems/muse_decode_layer/cand_r3_moe_lab.py'
    target.write_text(body)
    manifest=dict(job=rec['job'],core_job=core['job'],plan=plan,measured_minima=measured,native_tie_tolerance=.01,source=str(target.relative_to(ROOT)),sha256=hashlib.sha256(body.encode()).hexdigest(),base_sha256=hashlib.sha256(base.read_bytes()).hexdigest())
    (ROOT/'problems/muse_decode_layer/dev/moe_lab_dispatch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    return target

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('results',type=Path);args=a.parse_args()
    build(json.loads(args.results.read_text()))
