"""GPU0 direct arithmetic, cold graph comparisons and forced native glue coverage."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'problems/muse_decode_layer'))
sys.path.insert(0,str(ROOT/'problems/muse_moe_w4a16'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--compile90',action='store_true')
    args=ap.parse_args()
    path=ROOT/'problems/muse_moe_w4a16/cand_tma_grouped.py'
    tg=load('tma_candidate',path)
    if args.compile90:
        from torch.utils.cpp_extension import load_inline
        # Distinct cache name preserves the locally runnable module.
        declarations='''#include <torch/extension.h>
int64_t tg_capacity(int64_t,int64_t,int64_t);
void tg_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);
'''
        name='muse_tg_compile90_'+hashlib.sha256(tg.TG_CUDA.encode()).hexdigest()[:12]
        mod=load_inline(name=name,cpp_sources=declarations,cuda_sources=tg.TG_CUDA,
            functions=['tg_capacity','tg_run'],extra_cuda_cflags=tg._tg_cuda_flags((9,0)),verbose=False)
        print('TG_SM90_COMPILE',mod.__file__,flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32=False
    ref=load('real_layer_reference',ROOT/'problems/muse_decode_layer/reference.py')
    weights=ref.generate_weights(ref.WEIGHTS_SEED,device='cuda')
    mw={k:weights[k] for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs')}
    mw['post_norm_w']=weights['post_expert_norm']
    bank=tg._tg_prepare_bank(mw)
    native=tg._prepare_v9(mw)
    hum=tg._prepare_humming(mw)
    control={'native':native,'humming':hum,'dispatch':{128:'v9',256:'wide'}}
    choices=tg.TG_CHOICES if not args.quick else (tg.TG_CHOICES[0],)
    results=[];checks=0
    flush=torch.zeros((128<<20)//4,dtype=torch.float32,device='cuda')
    for m in (128,256):
        for mode in (('real',) if args.quick else ('real','matched_uniform','uniform')):
            datasets=[tg.make_fixture(weights,m,9000+7919*j+m,mode=mode) for j in range(3)]
            summary=tg.route_summary(datasets[0]);print('ROUTE',m,mode,summary,flush=True)
            refs=[tg._tg_reference(d,mw) for d in datasets]
            old=tg._TgGraph(lambda d:tg._kernel_v9(d,native),datasets[0])
            incumbent=tg._TgGraph(lambda d:tg._tg_incumbent_kernel(d,control),datasets[0])
            for choice in choices:
                tg._tg_configure(bank,m,choice)
                new=tg._TgGraph(lambda d,c=choice:tg._tg_run(d,bank,c),datasets[0])
                errors=[]
                for d,want in zip(datasets,refs):
                    new.refresh(d);out=new.replay();errors.append(tg._tg_check(out,want));checks+=1
                t0,t1,ratio,count=tg._tg_pair(old,new,datasets,flush,repeats=4 if args.quick else 20)
                ti,tn,ri,ci=tg._tg_pair(incumbent,new,datasets,flush,repeats=4 if args.quick else 20)
                row=dict(m=m,mode=mode,choice=choice,v9_us=t0,new_us=t1,ratio=ratio,
                    incumbent_us=ti,incumbent_ratio=ri,repeats=count,errors=errors,routes=summary)
                results.append(row);print('TG_BENCH',json.dumps(row),flush=True)
                del new
            del old,incumbent,refs,datasets
    # Different M/start, duplicates, no-local paths and a changed input on each replay.
    edge_m=(1,8,33,129,257) if not args.quick else (4,)
    for m in edge_m:
        d=tg.make_fixture(weights,m,1647+m,mode='real')
        for start,kind in ((0,'real'),(64,'offset'),(0,'empty'),(0,'allhot')):
            ids=d['topk_ids'].clone()
            ids=torch.where(ids>=0,ids+start,ids)
            if kind=='empty':ids.fill_(-1)
            if kind=='allhot':ids.fill_(start+3)
            fixture={**d,'topk_ids':ids,'config':{**d['config'],'local_start':start}}
            oracle=tg._tg_reference(fixture,mw)
            choice=choices[0];tg._tg_configure(bank,m,choice)
            graph=tg._TgGraph(lambda x:tg._tg_run(x,bank,choice),fixture)
            for _ in range(3):tg._tg_check(graph.replay(),oracle);checks+=1
            print('TG_EDGE',m,kind,'pass',flush=True)
    print('TG_VALIDATION',json.dumps(dict(arch=torch.cuda.get_device_capability(),checks=checks,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=results)),flush=True)


if __name__=='__main__':main()
