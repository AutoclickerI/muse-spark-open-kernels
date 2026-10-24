"""Execute every new family locally; separately compile the real Hopper specializations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'problems/muse_prefill_layer'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def compile90(g):
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    records=[]
    for m in (33,753,3012,16384):
        for mode in (0,1,2):
            for tile,(bm,bn,bk,warps,stages) in enumerate(g.GLUE_GEMMS):
                n=4096 if mode==0 else 8192;k=1024 if mode==1 else 8192 if mode==0 else 4096
                sig=dict(X='*bf16',W='*bf16',RAW='*bf16',OUT='*bf16',S='*fp32',NEW='*fp32',
                    POST='*fp32',ALPHA='*fp32',BETA='*fp32',PRE='*fp32',SQ='*fp32',COUNT='*i32')
                const=dict(M=m,N=n,K=k,MODE=mode,BM=bm,BN=bn,BK=bk)
                kernel=triton.compile(ASTSource(g._glue_gemm_norm,sig,const),target=GPUTarget('cuda',90,32),
                    options=dict(num_warps=warps,num_stages=stages))
                records.append(dict(role=mode,m=m,tile=tile,shared=kernel.metadata.shared,
                    wgmma='wgmma.mma_async' in kernel.asm['ptx']))
        for bm in (8,16,32):
            for sort in (False,True):
                sig=dict(X='*bf16',W='*bf16',BIAS='*fp32',LS='*i32',IDS='*i32',PROBS='*fp32',SLOT_E='*i32',SLOT_POS='*i32',COUNTS='*i32')
                const=dict(M=m,K=8192,BM=bm,BK=64,SORT=sort)
                kernel=triton.compile(ASTSource(g._glue_router_gemm,sig,const),target=GPUTarget('cuda',90,32),
                    options=dict(num_warps=4,num_stages=3))
                records.append(dict(role='router',m=m,bm=bm,sort=sort,shared=kernel.metadata.shared,
                    wgmma='wgmma.mma_async' in kernel.asm['ptx']))
    print('GLUE_COMPILE90',json.dumps(records),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('file',nargs='?',default=str(ROOT/'runs/screen/glue_prefill/work.py'))
    ap.add_argument('--compile90',action='store_true');ap.add_argument('--tokens',type=int,default=33)
    args=ap.parse_args();g=load('glue_candidate',Path(args.file))
    if args.compile90:compile90(g);return
    torch.backends.cuda.matmul.allow_tf32=False
    ref=load('prefill_glue_ref',ROOT/'problems/muse_prefill_layer/reference.py')
    weights=ref.generate_weights(ref.WEIGHTS_SEED,device='cuda')
    st=g.prepare_base(weights)
    st.update(glue_choices={},glue_scratch={},dense_choices={},sms=torch.cuda.get_device_properties('cuda').multi_processor_count,
        _glue_router_original=weights['router'])
    gen=torch.Generator(device='cuda').manual_seed(164532)
    results=[];counter_checks=0
    flush=torch.zeros(128<<20,device='cuda',dtype=torch.uint8)
    for role,choices in (('o',g.GLUE_BOUNDARY_CHOICES),('unpooler',g.GLUE_BOUNDARY_CHOICES),
                          ('router',g.GLUE_ROUTE_CHOICES),('pooler',g.GLUE_POOL_CHOICES)):
        run,check,refresh=g._glue_case(st,role,args.tokens,gen)
        old=g._ScoreGraph(lambda:run(('stock',)));check(old.output)
        for c in choices:
            new=g._ScoreGraph(lambda c=c:run(c));error=check(new.output)
            for j in range(6):refresh(j);new.replay();error=max(error,check(new.output))
            t0,t1,ratio,count=g._glue_pair(old,new,refresh,flush,20,float('inf'))
            for j in (1,2):refresh(j);new.replay();error=max(error,check(new.output))
            if c[0]=='gemm' and role!='router':
                counter=st['glue_scratch'][(role,args.tokens,c)]['counter']
                assert torch.equal(counter,torch.zeros_like(counter));counter_checks+=1
            row=dict(role=role,m=args.tokens,choice=c,stock_us=t0,trial_us=t1,ratio=ratio,rel=error,pairs=count)
            results.append(row);print('GLUE_TEST',json.dumps(row),flush=True)
            del new
        del old,run,check,refresh
    print('GLUE_VALIDATION',json.dumps(dict(arch=torch.cuda.get_device_capability(),results=results,
        counter_checks=counter_checks,sha256=hashlib.sha256(Path(args.file).read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':main()
