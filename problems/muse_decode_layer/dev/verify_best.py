"""Exercise best's complete Hopper glue on sm120, with portable attention launches.

CUDA source strings are also compiled for sm90a by local_eval. --cold-native
compiles every optional native module for sm90a during prepare before substituting
its device launch, so cold prepare includes real native compilation and all tuners.
Use fresh TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR/CUDA_CACHE_PATH for cold measurements.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import types
import torch
import triton
import triton.language as tl

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval


@triton.jit
def raw_combine(OP,MP,LP,OUT,VS,S:tl.constexpr,L:tl.constexpr,LB:tl.constexpr,SP:tl.constexpr):
    t=tl.program_id(0);b=t//L;j=t%L
    h=tl.arange(0,16);sp=tl.arange(0,SP);d=tl.arange(0,64)
    rr=(j%LB)*16+h;base=(b*(L//LB)+j//LB)*S
    m=tl.load(MP+(base+sp)[None,:]*(LB*16)+rr[:,None],sp[None,:]<S,-float('inf'))
    l=tl.load(LP+(base+sp)[None,:]*(LB*16)+rr[:,None],sp[None,:]<S,0.)
    mx=tl.max(m,1);w=tl.where(m==-float('inf'),0.,tl.exp2(m-mx[:,None]))
    den=tl.sum(w*l,1);acc=tl.zeros((16,64),tl.float32)
    for si in range(S):
        ws=tl.sum(tl.where(sp[None,:]==si,w,0.),1)
        o=tl.load(OP+((base+si)*(LB*16)+rr)[:,None]*128+(h//8)[:,None]*64+d[None,:])
        acc+=ws[:,None]*o
    tl.store(OUT+t*1024+h[:,None]*64+d[None,:],acc/den[:,None]*tl.load(VS))


def install(mod,cold_native):
    original_load_inline=mod.load_inline
    # Native Lt uses this GPU's real cuBLASLt. Only FA3 and Hopper attention
    # device launches need stand-ins; all Python tuners and calls are unchanged.
    fake=types.ModuleType('sgl_kernel.flash_attn')
    fake.get_scheduler_metadata=lambda **kwargs:None
    def flash(q,**kw):
        b=kw['page_table'].shape[0];l=kw['max_seqlen_q'];lb=min(l,4)
        window=kw['window_size'][0]+1 if kw['window_size'][0]>=0 else 0
        span=window+lb-1 if window else kw['page_table'].shape[1]*kw['k_cache'].shape[1]
        split=256 if window else max(512,min(4096,triton.next_power_of_2(span//64)))
        ns=triton.cdiv(span,split);rows=b*(l//lb)*ns*(lb*16);dev=q.device
        op=torch.empty((rows,128),device=dev,dtype=torch.float32)
        mp=torch.empty(rows,device=dev);lp=torch.empty_like(mp)
        out=torch.empty((b*l,16,64),device=dev,dtype=torch.bfloat16)
        seq=kw['cache_seqlens']-l
        ks=kw.get('k_descale',torch.ones((),device=dev)).reshape(-1)[:1]
        vs=kw.get('v_descale',torch.ones((),device=dev)).reshape(-1)[:1]
        mod._launch(mod._attn_split_kernel,(ns,b,l//lb),q.to(torch.bfloat16),kw['k_cache'],kw['v_cache'],kw['page_table'],seq,
                    ks,op,mp,lp,kw['page_table'].stride(0),kw['softmax_scale'],ns,
                    L=l,LB=lb,HQ=16,D=64,PAGE=kw['k_cache'].shape[1],WINDOW=window,SPLIT=split,BLOCK_N=64,
                    num_warps=4 if lb*16<=32 else 8,num_stages=2)
        raw_combine[(b*l,)](op,mp,lp,out,vs,S=ns,L=l,LB=lb,SP=triton.next_power_of_2(ns),num_warps=4)
        return out
    fake.flash_attn_with_kvcache=flash
    sys.modules['sgl_kernel.flash_attn']=fake
    mod._attn_port_hopper=lambda device:True
    mod._fa3_on_device=lambda device:True
    mod.ATTN_BACKEND='fa3'
    built=set();calls={}
    class PortablePort:
        def occupancy(self,length):return 1
        def max_cluster(self,length,cs):return mod._num_sms(torch.device('cuda')) if cs==1 else 0
        def attn_forward(self,*a):
            assert len(a)==28,len(a)
            qkv,pos,loc,pt,seq,kc,vc,ks,vs,freq,po,pm,counts,p1,dsmem,out,b,l,stride,hq,hkv,pages,ps,window,ns,cs,scale,stream=a
            calls['attn_main']=calls.get('attn_main',0)+1
            mod._layer_attn_ext().attn_forward(qkv,1,b*l*stride,0,pos,loc,pt,seq,kc,vc,ks,vs,freq,po,pm,out,
                b,l,stride,hq,hkv,pages,ps,window,ns,scale,stream)
        def xforward(self,*a):
            assert len(a)==25,len(a)
            qkv,pos,loc,pt,seq,kc,vc,ks,vs,freq,part,counts,out,b,hq,hkv,stride,pages,ps,window,ns,scale,st,extra,stream=a
            dev=torch.device('cuda');rows=b*(hkv//2)*ns*8*16
            po=torch.empty(rows*64,device=dev);pm=torch.empty(rows*2,device=dev)
            calls['xqa']=calls.get('xqa',0)+1
            mod._layer_attn_ext().attn_forward(qkv,1,b*8*stride,0,pos,loc,pt,seq,kc,vc,ks,vs,freq,po.data_ptr(),pm.data_ptr(),out,
                b,8,stride,hq,hkv,pages,ps,window,ns,scale,stream)
    def load_inline(**kw):
        if kw['cuda_sources'] not in (mod.ATTN_PORT_CUDA_SRC,mod.ATTN_X_CUDA,mod.ATTN_G_CUDA,mod.ATTN_P8_CUDA):
            return original_load_inline(**kw)
        tag=hashlib.sha256(kw['cuda_sources'].encode()).hexdigest()[:12]
        if cold_native and tag not in built:
            opts=dict(kw);opts['name']='best_native_'+tag
            opts['extra_cuda_cflags']=[f for f in opts['extra_cuda_cflags'] if not f.startswith('-gencode')]+['-gencode=arch=compute_90a,code=sm_90a']
            original_load_inline(**opts)
        built.add(tag)
        return PortablePort()
    mod.load_inline=load_inline
    original_prepare=mod.prepare
    def prepare(weights):
        state=original_prepare(weights)
        # Force every eligible attention setting on fresh inputs, then force
        # fused PREFIX on all official tests and scored graphs below.
        for l in (1,8):
            for window in (2048,0):
                qkv,data=mod._attn_port_synthetic(1,l,window,torch.device('cuda'),ctx=2305)
                for choice in mod._attn_port_candidates(l):
                    got=mod._attn_port_variant(qkv,data,state,choice)
                    mod._attention_store_agrees(qkv,data)
                    ref=mod._attention_reference(qkv,data)
                    rounded=mod._attention_reference(qkv,data,True)
                    ok,err,bound=mod._attention_agrees(got,ref,rounded)
                    assert ok,(choice,l,window,err,bound)
        state['attention_plan']={(b,l,w):('attn_main' if l==1 else 'xqa_warp',None if l==1 else ('warp',3,2))
            for b in (1,16,32) for l in (1,8) for w in (True,False)}
        state['prefix_plan']={t:('fused',True) for t in (1,8,16,32,128,256)}
        print('FORCED_HOPPER',json.dumps({'native_modules':len(built),'calls':calls}),flush=True)
        return state
    mod.prepare=prepare
    return calls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cold-native',action='store_true');args=ap.parse_args()
    original=local_eval.load_module
    def load(name,path):
        mod=original(name,path)
        if name=='submission':install(mod,args.cold_native)
        return mod
    local_eval.load_module=load
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/cand_r3_best.py'),'--no-baseline',
              '--json',str(ROOT/'problems/muse_decode_layer/dev/best_forced_local.json')]
    local_eval.main()


if __name__=='__main__':main()
