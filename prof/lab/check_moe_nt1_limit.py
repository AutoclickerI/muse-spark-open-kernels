"""Validate NT1 integer scale recoding and early stage release, including Hopper launch glue."""
import importlib.util
import json
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[2]


def load():
    p = ROOT/'prof/lab/moe_nt1_limit.py'
    spec = importlib.util.spec_from_file_location('nt1_limit_check', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Portable:
    def __init__(self, native, kind, sms):
        self.native, self.kind, self.sms = native, kind, sms
        self.smem_optin = 227328
        self.calls = []
    def max_smem_attr(self):
        return 227328
    def smem_needed(self, *args):
        return self.native.smem_needed(*args)
    def resident_blocks(self, nt, stages, m):
        return 1 if stages==3 else 2
    def run(self, pointers, m, start, stages, grid, stream):
        assert len(pointers)==(15 if self.kind=='trace' else 13)
        assert all(isinstance(p,int) and p>0 for i,p in enumerate(pointers) if not(self.kind=='trace' and i==11))
        assert start in (0,96) and grid in (132,264)
        assert stream==torch.cuda.current_stream().cuda_stream
        self.calls.append((m,start,stages,grid))
        actual_stages=1 if self.kind=='early' else min(stages,2)
        actual_grid=min(grid,self.sms*self.native.resident_blocks(1,actual_stages,m))
        self.native.run(pointers,m,start,actual_stages,actual_grid,stream)


def main():
    started=time.monotonic();m=load();device='cuda'
    native={kind:m.nt1_module(kind) for kind in ('recode','raw','scale16','early','lop3','trace','no_decode','no_mma')}
    previous=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    worst=0.;checks=0;forced=0
    try:
        for seed in (5353,13272,21191):
            w=m.rg_generate_weights(seed,device=device);w['post_norm_w']=w['post_expert_norm']
            # Keep source scales finite while exercising zeros, signs and subnormals.
            for name in ('w13','w2'):
                sf=w[name+'_sf'].view(torch.uint8)
                sf[3,:16,:8]=torch.arange(8,device=device,dtype=torch.uint8)
                sf[7,:16,:16]|=128
            bank=m.nt1_banks(w)
            for key in ('w13p','w2p'):
                got=bank['banks']['scale16'][key]
                ref=bank['banks']['recode'][key]
                assert torch.equal(got[...,:32768],ref[...,:32768])
                recovered=((got[...,32768:].contiguous().view(torch.int16).int()-0x7000)//16).byte()
                assert torch.equal(recovered,ref[...,32768:])
            print('BANK_PASS',seed,flush=True)
            for tokens in (1,8,16,32):
                fresh=[m.fx_make_fixture(w,tokens,9000+tokens+7919*j) for j in range(3)]
                fresh+=[dict(fresh[0],topk_ids=torch.full_like(fresh[0]['topk_ids'],-1)),
                        dict(fresh[0],hidden_states=torch.zeros_like(fresh[0]['hidden_states']))]
                wants=[m.nt1_fp32(d,w) for d in fresh]
                for kind in ('recode','raw','scale16','early','lop3','trace'):
                    stages=1 if kind=='early' else 2
                    graph=m.LabGraph(lambda d,k=kind,s=stages:m.nt1_call(d,bank,k,s),fresh[0])
                    for d,want in zip(fresh,wants):
                        graph.refresh(d);got=graph.replay();torch.cuda.synchronize()
                        error,row=m.nt1_error(got,want);worst=max(worst,error);checks+=1
                        assert bool(torch.isfinite(got).all()) and error<.01 and row<.05,(seed,tokens,kind,error,row)
                    shifted=dict(fresh[0],topk_ids=torch.where(fresh[0]['topk_ids']>=0,fresh[0]['topk_ids']+96,fresh[0]['topk_ids']),
                                 config=dict(fresh[0]['config'],local_start=96))
                    offset_graph=m.LabGraph(lambda d,k=kind,s=stages:m.nt1_call(d,bank,k,s),shifted)
                    got=offset_graph.replay();torch.cuda.synchronize();error,row=m.nt1_error(got,wants[0]);checks+=1
                    assert error<.01 and row<.05,(seed,tokens,kind,'start96',error,row)
                    print('PASS',seed,tokens,kind,'worst',worst,flush=True)
                if seed==5353:
                    # Force all Hopper stage/grid paths through portable native
                    # arithmetic; early's two-stage shared footprint exceeds SM120.
                    old_flag=m.NT1_HOPPER;old_loader=m.load_inline;old_modules=m.NT1_MODULES
                    portable={k:Portable(v,k,bank['sms']) for k,v in native.items()}
                    def fake_load(**kwargs):
                        assert '-gencode=arch=compute_90a,code=sm_90a' in kwargs['extra_cuda_cflags']
                        assert kwargs['no_implicit_headers'] and not kwargs['with_pytorch_error_handling']
                        return next(p for k,p in portable.items() if kwargs['name'].startswith('muse_nt1_limit_'+k))
                    m.NT1_HOPPER=True;m.load_inline=fake_load;m.NT1_MODULES={}
                    try:
                        for kind in native:m.nt1_module(kind)
                        fake_bank=dict(bank,sms=132)
                        d=fresh[0];want=wants[0]
                        for kind in ('recode','raw','scale16','early','lop3','trace'):
                            for stages in (1,2,3):
                                for grid in (132,264):
                                    g=m.LabGraph(lambda d,k=kind,s=stages,gg=grid:m.nt1_call(d,fake_bank,k,s,gg),d)
                                    got=g.replay();torch.cuda.synchronize();err,row=m.nt1_error(got,want)
                                    assert err<.01 and row<.05,(kind,stages,grid,err,row)
                                    forced+=2
                    finally:m.NT1_HOPPER=old_flag;m.load_inline=old_loader;m.NT1_MODULES=old_modules
            fresh=wants=bank=w=None
        result=dict(native_graph_checks=checks,forced_hopper_calls=forced,worst_fp32=worst,
                    public_weight_seeds=3,scale_sign_zero_subnormal=True,seconds=time.monotonic()-started,
                    early_native_stages=1,early_hopper_stages='1/2/3 glue forced with stage1 arithmetic stand-in')
        (ROOT/'runs/screen/nt1_limit_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:torch.backends.cuda.matmul.allow_tf32=previous


if __name__=='__main__':main()
