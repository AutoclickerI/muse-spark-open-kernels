"""Force Hopper Python adapters with portable grouped GEMMs on sm120.

The canonical expansion, gathers, slot layout, epilogues and raw17 kernels run
unchanged. Only DeepGEMM/Humming native GEMMs and their kernel compilation use
portable stand-ins. Actual Hopper safety remains the separate H200 lab's job.
"""
from pathlib import Path
import argparse,json,sys,types,time
import torch
import triton
import triton.language as tl
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev')]
import local_eval
import verify_best


@triton.jit
def grouped(A,W,C,COUNTS,GAIN,N:tl.constexpr,M:tl.constexpr,PSUM:tl.constexpr,STRIDE:tl.constexpr,SCALE:tl.constexpr):
    pn=tl.program_id(0);pm=tl.program_id(1);e=tl.program_id(2)
    if PSUM:
        prior=tl.load(COUNTS+e-1,e>0,0);start=tl.cdiv(prior,128)*128
        count=tl.load(COUNTS+e)-start
    else:
        start=e*STRIDE;count=tl.load(COUNTS+e)
    if pm*16>=count:return
    rm=pm*16+tl.arange(0,16);rn=pn*64+tl.arange(0,64);rk=tl.arange(0,64)
    acc=tl.zeros((16,64),tl.float32)
    for kb in range(64):
        av=tl.load(A+(start+rm)[:,None].to(tl.int64)*4096+(kb*64+rk)[None,:],rm[:,None]<count,0)
        bv=tl.load(W+(e.to(tl.int64)*N+rn)[None,:]*4096+(kb*64+rk)[:,None])
        acc=tl.dot(av,bv,acc)
    if SCALE:acc*=tl.load(GAIN+e)
    tl.store(C+(start+rm)[:,None].to(tl.int64)*N+rn[None,:],acc.to(tl.bfloat16),rm[:,None]<count)


def install(mod,cold):
    calls=verify_best.install(mod,cold)
    holder={};native_calls={'dg':0,'humming':0};record={'native_calls':native_calls}
    original_build=mod._best6_build_adapters
    mod._best6_hopper=lambda device:True
    fake=types.ModuleType('deep_gemm')
    fake.use_deterministic_algorithms=lambda enabled:None
    def dg(a,w,c,psum,use_psum_layout,**kwargs):
        assert use_psum_layout and not kwargs,kwargs
        m=(a.shape[0]-4096)//8;n=w.shape[1]
        grouped[(n//64,triton.cdiv(m,16),32)](a,w,c,psum,psum,N=n,M=m,PSUM=True,STRIDE=0,SCALE=False,num_warps=4)
        native_calls['dg']+=1
    fake.m_grouped_bf16_gemm_nt_contiguous=dg
    sys.modules['deep_gemm']=fake
    original_import=mod._best6_import_humming
    def humming_import():
        original_import()
        class Kernel:
            @staticmethod
            def prepare_kernels(meta,compute,settings):
                assert json.loads(compute)['use_f16_accum'] is False
                return torch.tensor([[i,i+1,i,1] for i in range(len(settings))],dtype=torch.int64)
        mod._pf_HummingKernel=Kernel
        # Retain the real schema/transform/heuristics Python code; only native
        # compilation and launch are substituted on this unsupported GPU.
        class Ops:
            @staticmethod
            def launch_kernel(*,configs,inputs,outputs,locks,expert_layout,valid_shape_m,weight,weight_scale,weight_scale_2):
                assert configs.device.type=='cpu' and configs.numel()==4
                st=holder['state'];n=outputs.shape[-1];m=inputs.shape[0]//32
                assert valid_shape_m in (128,256) and m in (128,256)
                name='w13' if n==8192 else 'w2';bank=st[name+'p']
                scratch=torch.empty((32,n,4096),device=inputs.device,dtype=torch.bfloat16)
                mod._pf_canonical_expand_coalesced[(32,n//16,4)](bank,bank.view(torch.int32),scratch.view(torch.int32),expert_layout,
                    ROWS=n,GATED=n==8192,KB=1024,num_warps=8)
                gain=st['gs13_gate' if n==8192 else 'gs2']
                grouped[(n//64,triton.cdiv(m,16),32)](inputs,scratch,outputs,expert_layout,gain,
                    N=n,M=m,PSUM=False,STRIDE=m,SCALE=True,num_warps=4)
                native_calls['humming']+=1
        mod._pf_humming_ops=Ops
    mod._best6_import_humming=humming_import
    def build(weights):
        state=original_build(weights);holder['state']=state
        assert all(n in state['available'] for n in ('raw17','dg_w8','humming_masked')),state['rejected']
        return state
    mod._best6_build_adapters=build
    original_prepare=mod.prepare
    def prepare(weights):
        started=time.monotonic();state=original_prepare(weights)
        # All adapters must survive their independent gates and fresh graph
        # replays; the full layer below forces them at the target M as well.
        for m in (128,256):
            row=state['moe6_rows'][m]
            assert row['reason']=='measured',row
            for name in mod._BEST6_CHOICES:
                assert 'mean_us' in row['choices'].get(name,{}),(m,name,row)
                assert name not in row['rejected'],(m,name,row)
        cycle={'calls':0,'choices':{m:{n:0 for n in ('raw17','humming_masked','dg_w8')} for m in (128,256)}}
        record['cycle']=cycle
        original_experts=mod._best6_experts
        def experts(pr,x,wts,lids,st):
            if x.shape[0] in (128,256):
                name=('raw17','humming_masked','dg_w8')[cycle['calls']%3]
                cycle['calls']+=1;cycle['choices'][x.shape[0]][name]+=1
                return mod._best6_optional(pr,wts,lids,st,name)
            return original_experts(pr,x,wts,lids,st)
        mod._best6_experts=experts
        print('BEST6_FORCED_READY',json.dumps(dict(native=native_calls,extra_s=time.monotonic()-started)),flush=True)
        return state
    mod.prepare=prepare
    return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cold-native',action='store_true');args=ap.parse_args()
    old=local_eval.load_module;records={}
    def load(name,path):
        m=old(name,path)
        if name=='submission':records['forced']=install(m,args.cold_native)
        return m
    local_eval.load_module=load
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/cand_r3_best6.py'),'--no-baseline',
              '--json',str(ROOT/'problems/muse_decode_layer/dev/best6_forced_local.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:
            record=records['forced']
            assert all(v>0 for row in record['cycle']['choices'].values() for v in row.values()),record
            (Path(__file__).parent/'forced_families_local.json').write_text(json.dumps(record,indent=2)+'\n')
            print('BEST6_FORCED_COMPLETE',json.dumps(record),flush=True)
        raise
if __name__=='__main__':main()
