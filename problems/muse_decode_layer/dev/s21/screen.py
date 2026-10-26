"""S21 layer screens: real allocation/launch glue with explicit Hopper arithmetic stand-ins."""
import importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('s21_prior_screen',H.parent/'s20/screen.py');prior=importlib.util.module_from_spec(sp);sp.loader.exec_module(prior)
import sys
R=H.parents[3];FULL=R/'problems/muse_attention_decode/dev/full_attention'
sys.path[:0]=[str(R/'tools'),str(FULL)]
import chain_standins

def install(m,forced=False):
    records=prior.install(m,forced)
    if hasattr(m,'_p5b_merge_setup'):
        def setup():
            import hashlib
            src=m._P5B_REDUCE_SRC
            m._P5B_REDUCE=m.load_inline(name='p5_chain_cand_reduce_'+hashlib.sha256(src.encode()).hexdigest()[:12],
                cpp_sources='void row_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);',
                cuda_sources=src,functions=['row_forward'],extra_cuda_cflags=['-O3'])
            m._P5B_BRIDGE='explicit local FA3 main stand-in'
        m._p5b_merge_setup=setup
        def main(q,data,ns,fo,lse,oa,la,meta):
            chain_standins.fa_main(m,q,data,ns,oa,la)
            if ns==1:fo.copy_(oa[0].permute(1,0,2).to(m.torch.bfloat16))
        m._p5b_merge_main=main
    if hasattr(m,'_P5B_VQ_SOURCES'):
        cache={}
        def module(role):
            if role not in cache:cache[role]=chain_standins.native(m,m._P5B_VQ_SOURCES[role])
            native=cache[role]
            class Wrapped:
                def vq8_forward(self,*args):
                    args=list(args);args[22]=2
                    return native.vq8_forward(*args)
            return Wrapped()
        m._p5b_vq_module=module
    if hasattr(m,'NP_CONFIG'):
        m.NP_LOCAL_MM=prior.portable.mm
        prepare=m.np_prepare
        def np_prepare(state):
            bank=prepare(state)
            for role,(wp,_) in bank['banks'].items():
                n,k=state[role].shape;bw=m.NP_CONFIG[role][1]
                recovered=wp.reshape(k//64,n//bw,bw,64).permute(1,2,0,3).reshape(n,k)
                assert m.torch.equal(recovered.view(m.torch.int16),state[role].view(m.torch.int16)),role
            return bank
        m.np_prepare=np_prepare
    return records
if __name__=='__main__':prior.s19.old.install=install;prior.s19.old.main()
