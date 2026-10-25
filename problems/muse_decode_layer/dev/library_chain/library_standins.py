"""Local ABI screen for VDSK tiles exceeding sm120's 99KiB shared memory."""
import local_standins


def install(m):
    result=local_standins.install(m)
    if m.torch.cuda.get_device_capability()==(9,0):return result
    if not hasattr(m,'LIBRARY_NATIVE_ROWS'):return result
    real=m._vdsk_gemm
    class Kernel:
        def __getitem__(self,grid):
            def launch(X,W,P,xd,wd,M,N,K,S,**kw):
                if not kw.get('CHAIN',False):return real[grid](X,W,P,xd,wd,M,N,K,S,**kw)
                assert kw['launch_pdl'] and tuple(P.shape)==(S,M,N)
                assert grid==(m.triton.cdiv(M,kw['BT'])*(N//kw['BW']),S)
                assert bool(xd is not None)==bool(kw['TMA'])
                assert bool(wd is not None)==bool(kw['TMA'])
                # Same split bounds and FP32 partial ABI; only the unsupported
                # device launch is substituted. The real Python wrapper runs.
                steps=K//kw['BK']
                for ps in range(S):
                    lo=(ps*steps//S)*kw['BK'];hi=((ps+1)*steps//S)*kw['BK']
                    m.torch.mm(X[:,lo:hi],W[:,lo:hi].T,out_dtype=m.torch.float32,out=P[ps])
            return launch
    m._vdsk_gemm=Kernel()
    result['library_vdsk']='CHAIN-only launch substitute: exact K partitions and FP32 partial ABI; Hopper tile >sm120 shared limit'
    return result
