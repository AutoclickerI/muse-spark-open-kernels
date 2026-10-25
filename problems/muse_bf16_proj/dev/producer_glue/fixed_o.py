"""Exact S11 config6 M1 O body with a boundary epilogue, no weight repack."""
FIXED_O_CUDA = __CUDA__
_FO_MODULE=[]


def fixed_o_module():
    if not _FO_MODULE:
        from torch.utils.cpp_extension import load_inline
        import hashlib
        arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
        tag=hashlib.sha256(FIXED_O_CUDA.encode()).hexdigest()[:12]
        cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nvoid run(std::vector<int64_t>,int64_t,int64_t,int64_t);'
        _FO_MODULE.append(load_inline(name=f'pg_fixed_o_{arch}_{tag}',cpp_sources=cpp,cuda_sources=FIXED_O_CUDA,
            functions=['run'],no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'],verbose=False))
    return _FO_MODULE[0]


class FixedOBoundary:
    def __init__(self,w,t,s,role,stages=4,emit_meta=True,fuse=True,trace=False,**kw):
        assert t==1 and tuple(w.shape)==(8192,1024) and role=='b1'
        self.w,self.t,self.s,self.role=w,t,1,role
        self.n,self.k,self.bn=8192,1024,16
        self.stages=stages;self.fuse,self.trace=fuse,trace
        self.ctas=512;self.joiners=512
        dev=w.device
        self.p=torch.empty((1,1,8192),device=dev,dtype=torch.bfloat16)
        self.count=torch.zeros(1,device=dev,dtype=torch.uint64)
        self.gen=torch.zeros(512,device=dev,dtype=torch.uint64)
        self.stamps=torch.empty((512,3),device=dev,dtype=torch.uint64)
        self.a=torch.empty(8192,device=dev);self.b=torch.empty_like(self.a)
        self.stats=torch.empty((512,4),device=dev)
        self.mod=fixed_o_module()

    def __call__(self,x,state,stream=None,start=None):
        out=torch.empty((1,8192),device=x.device,dtype=torch.bfloat16)
        sout=torch.empty((1,8192),device=x.device,dtype=torch.float32)
        v=[self.w,x,self.p,self.stamps,self.count,self.gen,stream,sout,out,self.a,self.b,self.stats,
           state['alpha1'],state['beta1'],state['pre_ffn_w']]
        self.mod.run([a.data_ptr() for a in v],int(self.fuse),int(self.trace),torch.cuda.current_stream().cuda_stream)
        if not self.fuse:return self.p,1
        return out,sout


def ProducerHybrid(w,t,s,role,**kw):
    if t==1 and role=='b1':return FixedOBoundary(w,t,s,role,**kw)
    return ProducerStats(w,t,s,role,**kw)
