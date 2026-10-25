import ctypes as _brct
import hashlib as _brhash
import os as _bros
import torch
from torch.utils.cpp_extension import load_inline as _brload

# rows, K elements/stage, stages, explicit K splits, CTAs per SM.
BR_CONFIGS={
 0:(64,64,4,1,1),1:(64,64,8,1,1),2:(64,128,4,1,1),3:(64,128,8,1,1),4:(64,256,4,1,1),
 5:(64,64,4,2,1),6:(64,64,8,2,1),7:(64,128,4,2,1),8:(64,128,4,4,1),9:(64,64,8,4,1),
 10:(64,64,8,8,1),11:(64,64,4,1,2),12:(64,64,8,1,2),13:(64,128,4,2,2),
 14:(32,128,4,1,1),15:(32,128,8,2,1),16:(128,32,8,1,1),17:(128,64,4,2,1),
 18:(128,64,8,4,1),19:(64,256,4,2,1),20:(64,128,8,4,1)}
_BR_LIB=None

def br_swizzle(packed):
    """Self-inverse row XOR, preserving every BF16 bit."""
    ks,rows,uk,r,kc=packed.shape
    if kc<64:return packed
    index=torch.arange(kc,device=packed.device).unsqueeze(0)^((torch.arange(r,device=packed.device)&1).unsqueeze(1)*32)
    return packed.gather(-1,index.view(1,1,1,r,kc).expand(ks,rows,uk,r,kc)).contiguous()

def br_library():
    global _BR_LIB
    if _BR_LIB is None:
        old=_bros.environ.get('TORCH_CUDA_ARCH_LIST')
        _bros.environ['TORCH_CUDA_ARCH_LIST']='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        try:
            path=_brload(name='muse_bulk_ring_'+_brhash.sha256(BR_CUDA.encode()).hexdigest()[:16],cpp_sources='',
                cuda_sources=BR_CUDA,functions=None,is_python_module=False,no_implicit_headers=True,
                extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','--ptxas-options=-v'],verbose=False)
        finally:
            if old is None:_bros.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_bros.environ['TORCH_CUDA_ARCH_LIST']=old
        _BR_LIB=_brct.CDLL(path)
        _BR_LIB.br_run.argtypes=[_brct.c_void_p]*4+[_brct.c_int]*11+[_brct.c_void_p]
        _BR_LIB.br_run.restype=_brct.c_int
    return _BR_LIB

class BulkRingProjection:
    """Lossless tiled BF16 bank; persistent bulk-copy ring; consumer partial ABI."""
    def __init__(self,w,tokens,config,pdl=True,trace=False,packed=None,grid=None):
        self.r,self.kc,self.st,self.ks,self.cta=BR_CONFIGS[config]
        if w.dtype!=torch.bfloat16 or w.ndim!=2 or not w.is_contiguous() or w.shape[0]%self.r or w.shape[1]%(self.kc*self.ks):
            raise ValueError('invalid original BF16 bank')
        if tokens<1 or tokens>32:raise ValueError('bulk ring tokens outside 1..32')
        self.n,self.k=w.shape;self.tokens=tokens;self.config=config;self.pdl=pdl;self.lib=br_library()
        prop=torch.cuda.get_device_properties(w.device)
        self.grid=grid or prop.multi_processor_count*self.cta
        self.reserve=self.st*self.r*self.kc*2+self.st*8
        if prop.major==9:self.reserve=max(self.reserve,(120 if self.cta==1 else 80)*1024)
        rows=self.n//self.r;uk=self.k//self.ks//self.kc
        if rows*uk*self.ks<self.grid:raise ValueError('insufficient weight tiles for equal grid')
        local=(self.grid+rows*self.ks-1)//(rows*self.ks)+1
        self.splits=self.ks*local
        self.w=(br_swizzle(w.view(rows,self.r,self.ks,uk,self.kc).permute(2,0,3,1,4).contiguous()) if packed is None else packed)
        self.out=torch.empty((self.splits,tokens,self.n),device=w.device,dtype=torch.float32)
        self.stamps=torch.empty((self.grid,5),device=w.device,dtype=torch.int64) if trace else None
        # Resolve attributes outside capture; reserve enough Hopper smem to cap
        # residency at exactly one/two CTAs per SM for a full fixed grid.
        error=self.lib.br_run(0,0,0,0,tokens,self.n,self.k,self.r,self.kc,self.st,self.ks,self.splits,0,
            self.reserve,int(pdl),torch.cuda.current_stream(w.device).cuda_stream)
        if error:raise RuntimeError(f'br_prepare CUDA={error}')

    def __call__(self,x):
        if x.dtype!=torch.bfloat16 or not x.is_contiguous() or x.shape!=(self.tokens,self.k):raise ValueError('activation ABI mismatch')
        error=self.lib.br_run(self.w.data_ptr(),x.data_ptr(),self.out.data_ptr(),self.stamps.data_ptr() if self.stamps is not None else 0,
            self.tokens,self.n,self.k,self.r,self.kc,self.st,self.ks,self.splits,self.grid,self.reserve,int(self.pdl),torch.cuda.current_stream(x.device).cuda_stream)
        if error:raise RuntimeError(f'br_run CUDA={error}')
        return self.out,self.splits
