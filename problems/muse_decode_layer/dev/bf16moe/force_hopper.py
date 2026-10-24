"""Run the full official local screen with Hopper-only BF16 glue forced on sm120.

Only the Hopper grouped GEMM is replaced, by a BF16 Triton grouped implementation.
All routing, streams, V9, epilogues, prepare admission and graph replay stay intact.
"""
from pathlib import Path
import sys
import torch
import triton
import triton.language as tl
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval


@triton.jit
def portable_grouped(A,W,O,COUNTS,ENDS,MASKED:tl.constexpr,STRIDE:tl.constexpr,
                     N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    e=tl.program_id(0)
    base=tl.program_id(1)*BM
    col=tl.program_id(2)*BN+tl.arange(0,BN)
    count=tl.load(COUNTS+e)
    if base<count:
        if MASKED:
            start=e*STRIDE
        else:
            start=tl.load(ENDS+e-1,e>0,0)
        row=base+tl.arange(0,BM)
        k=tl.arange(0,BK)
        acc=tl.full((BM,BN),0,tl.float32)
        for kb in range(tl.cdiv(K,BK)):
            kk=kb*BK+k
            aa=tl.load(A+(start+row[:,None])*K+kk[None,:],row[:,None]<count,0)
            bb=tl.load(W+e*N*K+col[None,:]*K+kk[:,None])
            acc+=tl.dot(aa,bb)
        tl.store(O+(start+row[:,None])*N+col[None,:],acc,row[:,None]<count)


class PortableGrouped:
    def __init__(self):
        self.sms=torch.cuda.get_device_properties(0).multi_processor_count

    def get_num_sms(self):return self.sms

    def set_num_sms(self,sms):self.sms=sms

    def m_grouped_bf16_gemm_nt_masked(self,a,w,out,counts,expected):
        groups,m,k=a.shape;n=w.shape[1]
        portable_grouped[(groups,triton.cdiv(m,16),triton.cdiv(n,64))](a,w,out,counts,counts,
                       MASKED=True,STRIDE=m,N=n,K=k,num_warps=4)

    def m_grouped_bf16_gemm_nt_contiguous(self,a,w,out,ends,use_psum_layout=True):
        # The stand-in receives the same GPU prefix sums. Recover counts from
        # the true per-call routing metadata supplied by the dispatch hook.
        m=(a.shape[0]-4096)//8;n=w.shape[1];k=w.shape[2]
        portable_grouped[(32,triton.cdiv(m,16),triton.cdiv(n,64))](a,w,out,self.counts,ends,
                       MASKED=False,STRIDE=0,N=n,K=k,num_warps=4)


original_load=local_eval.load_module


def forced_load(name,path):
    mod=original_load(name,path)
    if name!='submission':return mod
    backend=PortableGrouped()
    mod._bf_is_hopper=lambda:True
    mod._bf_backend=lambda:backend
    original_dense=mod._bf_dense
    def dense(a,w,out,counts,ends,policy,bank):
        backend.counts=counts
        return original_dense(a,w,out,counts,ends,policy,bank)
    mod._bf_dense=dense
    original_prepare=mod.prepare
    def prepare(weights):
        state=original_prepare(weights)
        bank=mod._bf_make_bank(weights);bank['dg']=backend
        state['bf16_moe_bank']=bank
        for m,threshold in ((32,16),(256,32)):
            histograms=[]
            for j in range(3):
                data=mod._bf_fixture_make_fixture(weights,m,5353+m+7919*j)
                ids=data['topk_ids']
                histograms.append(torch.bincount(ids[ids>=0],minlength=32).cpu().tolist())
            assert mod._bf_bmm_plan(bank,m,threshold,histograms,compact=True)
        state['bf16_moe_choices']={32:('bmm_tn_f32',16,.25,32,0,1),
                                 128:('masked',32,.25,96,32,1),
                                 256:('bmm_nn_f32',32,.5,160,0,1)}
        print('FORCED_HOPPER_BF16_GROUPED_STANDIN',state['bf16_moe_choices'],flush=True)
        return state
    mod.prepare=prepare
    return mod


if __name__=='__main__':
    local_eval.load_module=forced_load
    sys.argv=['local_eval.py',str(ROOT/'problems/muse_decode_layer/cand_r3_bf16moe.py'),
              '--json',str(ROOT/'runs/screen/bf16-forced-hopper.json')]+sys.argv[1:]
    local_eval.main()
