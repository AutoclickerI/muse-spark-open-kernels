"""Validate fixed-phase MoE math, metadata ABI, graphs and every Hopper launch path."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch

ROOT=Path(__file__).resolve().parents[2]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Portable:
    def __init__(self,native,kind,sms):
        self.native,self.kind,self.sms=native,kind,sms
        self.smem_optin=227328;self.calls=[]
    def max_smem_attr(self):return 227328
    def smem_needed(self,*args):return self.native.smem_needed(*args)
    def resident_blocks(self,nt,st,m):
        return (4 if st<3 else 3) if self.kind=='K2' else (2 if st<3 else 1)
    def run(self,p,m,start,st,g,stream):
        assert len(p)==(15 if self.kind=='K2' else 13 if self.kind in ('INC','R2','L26') else 14)
        assert all(isinstance(v,int) and v>0 for v in p)
        assert start in (0,96) and g in (132,264,396,528)
        assert stream==torch.cuda.current_stream().cuda_stream
        self.calls.append((m,start,st,g))
        actual_st=st if self.kind in ('K2','D','ABCD') else min(st,2)
        actual_g=min(g,self.sms*self.native.resident_blocks(1,actual_st,m))
        self.native.run(p,m,start,actual_st,actual_g,stream)


def check_meta(meta,d):
    v=meta.view(torch.int32).cpu();ids=d['topk_ids'].long().cpu()-d['config']['local_start'];m=len(ids)
    valid=(ids>=0)&(ids<32);flat=ids.flatten();cnt=torch.bincount(flat[valid.flatten()],minlength=32)
    n=int(cnt.sum());off=torch.cat((torch.zeros(1,dtype=torch.int64),cnt.cumsum(0)))
    nb=(cnt+7)//8;us=torch.cat((torch.zeros(1,dtype=torch.int64),nb.cumsum(0)))
    assert torch.equal(v[:4].long(),torch.tensor([1,m,d['config']['local_start'],n]))
    assert torch.equal(v[4:36].long(),cnt)
    assert torch.equal(v[36:69].long(),off)
    assert torch.equal(v[69:101].long(),nb)
    assert torch.equal(v[101:134].long(),us*512) and torch.equal(v[134:167].long(),us*256)
    assert torch.equal(v[167:167+m].long(),valid.sum(1))
    order=flat.argsort(stable=True);order=order[valid.flatten()[order]]
    assert torch.equal(v[199:199+n].long(),order)
    positions=torch.full_like(flat,-1);positions[order]=torch.arange(n)
    assert torch.equal(v[455:455+m*8].long(),positions)


def main():
    begin=time.monotonic();m=load('prof/lab/moe_nt1_fixed.py','fixed_check')
    modules={k:m.nt1_module(k) for k in m.NT1_SOURCES}
    prev=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    gates=stage_checks=meta_checks=forced=0;worst=against_inc=0.
    try:
        for seed in (5353,13272,21191):
            w=m.rg_generate_weights(seed);w['post_norm_w']=w['post_expert_norm']
            for name in ('w13','w2'):
                sf=w[name+'_sf'].view(torch.uint8)
                sf[3,:16,:8]=torch.arange(8,device='cuda',dtype=torch.uint8)
                sf[7,:16,:16]|=128
            state=m.nt1_banks(w)
            for key in ('w13p','w2p'):
                base=state['banks']['INC'][key];micro=state['banks']['D'][key];half=state['banks']['K2'][key]
                assert base.numel()==micro.numel()==half.numel() and base.dtype==micro.dtype==half.dtype==torch.uint8
                e,t=base.shape[:2]
                wb=micro[...,:4096].reshape(e,t,8,8,32,16).permute(0,1,3,2,4,5).reshape(e,t,32768)
                sb=micro[...,4096:].reshape(e,t,8,8,32,2).permute(0,1,3,4,2,5).reshape(e,t,4096)
                assert torch.equal(wb,base[...,:32768]) and torch.equal(sb,base[...,32768:])
            for rows in (1,8,16,32):
                ds=[m.fx_make_fixture(w,rows,9000+rows+7919*j) for j in range(3)]
                all_local=dict(ds[0],topk_ids=torch.tensor([3,7,8,12,15,20,24,31],device='cuda',dtype=torch.int32)[None].expand(rows,-1).contiguous())
                ds += [all_local,dict(ds[0],topk_ids=torch.full_like(ds[0]['topk_ids'],-1)),
                       dict(ds[0],hidden_states=torch.zeros_like(ds[0]['hidden_states']))]
                wants=[m.nt1_fp32(d,w) for d in ds]
                refs=[m.nt1_call(d,state,'INC',scratch=True) for d in ds];torch.cuda.synchronize()
                for kind in m.NT1_SOURCES:
                    stages=3 if kind=='K2' else 2;grid=528 if kind=='K2' else 264
                    graph=m.LabGraph(lambda d,k=kind,s=stages,g=grid:m.nt1_call(d,state,k,s,g),ds[0])
                    for d,want,ref in zip(ds,wants,refs):
                        graph.refresh(d);got=graph.replay();torch.cuda.synchronize()
                        err,row=m.nt1_error(got,want);ei,ri=m.nt1_error(got,ref[0]);worst=max(worst,err);against_inc=max(against_inc,ei)
                        assert bool(torch.isfinite(got).all()) and err<.01 and row<.05,(seed,rows,kind,err,row)
                        assert ei<.01 and ri<.05,(seed,rows,kind,'incumbent',ei,ri);gates+=1
                        debug=m.nt1_call(d,state,kind,stages,grid,scratch=True);torch.cuda.synchronize()
                        p=int(((d['topk_ids']>=0)&(d['topk_ids']<32)).sum())
                        if kind!='K2':
                            assert torch.equal(debug[1][:p],ref[1][:p]),(seed,rows,kind,'SwiGLU')
                            assert torch.equal(debug[2][:p],ref[2][:p]),(seed,rows,kind,'W2');stage_checks+=2
                        if kind in ('A','ABC','ABCD','ABCD128'):check_meta(debug[4],d);meta_checks+=1
                        assert not torch.count_nonzero(state['tickets'])
                        if kind in ('B','ABC','ABCD','ABCD128'):assert not torch.count_nonzero(state['counters'][kind])
                    shifted=dict(ds[0],topk_ids=torch.where(ds[0]['topk_ids']>=0,ds[0]['topk_ids']+96,ds[0]['topk_ids']),config=dict(ds[0]['config'],local_start=96))
                    graph2=m.LabGraph(lambda d,k=kind,s=stages,g=grid:m.nt1_call(d,state,k,s,g),shifted)
                    got=graph2.replay();torch.cuda.synchronize();err,row=m.nt1_error(got,wants[0]);gates+=1
                    assert err<.01 and row<.05,(seed,rows,kind,'offset96',err,row)
                    print('NATIVE',seed,rows,kind,'FP32',err,flush=True)
                if seed==5353:
                    old_flag,old_load,old_mods=m.NT1_HOPPER,m.load_inline,m.NT1_MODULES
                    portable={k:Portable(v,k,state['sms']) for k,v in modules.items()}
                    names={'muse_nt1_limit_'+k+hashlib.sha256(m.NT1_SOURCES[k].encode()).hexdigest()[:12]:v for k,v in portable.items()}
                    def fake_load(**kw):
                        assert '-gencode=arch=compute_90a,code=sm_90a' in kw['extra_cuda_cflags']
                        assert kw['no_implicit_headers'] and not kw['with_pytorch_error_handling']
                        return names[kw['name']]
                    m.NT1_HOPPER=True;m.load_inline=fake_load;m.NT1_MODULES={}
                    try:
                        for k in modules:m.nt1_module(k)
                        fake_state=dict(state,sms=132)
                        for k in modules:
                            for st in (1,2,3):
                                for grid in (132,264,528):
                                    g=m.LabGraph(lambda d,k=k,s=st,g=grid:m.nt1_call(d,fake_state,k,s,g),ds[0])
                                    got=g.replay();torch.cuda.synchronize();e,r=m.nt1_error(got,wants[0])
                                    assert e<.01 and r<.05,(k,st,grid,e,r);forced+=2
                    finally:m.NT1_HOPPER=old_flag;m.load_inline=old_load;m.NT1_MODULES=old_mods
        result=dict(native_graph_gates=gates,exact_stage_comparisons=stage_checks,metadata_checks=meta_checks,
            forced_hopper_calls=forced,worst_fp32=worst,worst_vs_incumbent=against_inc,seconds=time.monotonic()-begin,
            expert_counters_reset=True,raw_u8_scales=True,bank_bytes_unchanged=True,
            portable_depth='standard and WS128 stages3 map to2; WS64 and K2 execute native stages1/2/3')
        (ROOT/'runs/screen/nt1_fixed_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
    finally:torch.backends.cuda.matmul.allow_tf32=prev


if __name__=='__main__':main()
