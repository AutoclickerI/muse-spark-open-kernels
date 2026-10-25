import ctypes as _vdct
import hashlib as _vdhash
import os as _vdos
import torch
from torch.utils.cpp_extension import load_inline as _vdload

_VD_LIBS={}


def vd_library(kind):
    if kind in _VD_LIBS:return _VD_LIBS[kind]
    source={'lt':VD_LT_CUDA,'incumbent':VD_INC_CUDA,'cutlass':VD_CUTLASS_HOST}[kind]
    if kind=='cutlass':
        import base64,zlib,tempfile
        raw=zlib.decompress(base64.b64decode(VD_HOST_BINARY))
        if _vdhash.sha256(raw).hexdigest()!=VD_HOST_SHA256:raise RuntimeError('CUTLASS host binary hash')
        with tempfile.NamedTemporaryFile(suffix='.so') as handle:
            handle.write(raw);handle.flush();lib=_vdct.CDLL(handle.name)
        lib.vd_cutlass.argtypes=[_vdct.c_void_p]*4+[_vdct.c_size_t]+[_vdct.c_int]*6+[_vdct.c_void_p,_vdct.POINTER(_vdct.c_size_t),_vdct.c_int,_vdct.c_void_p]
        lib.vd_cutlass.restype=_vdct.c_int
        _VD_LIBS[kind]=lib;return lib
    arch='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
    previous=_vdos.environ.get('TORCH_CUDA_ARCH_LIST');_vdos.environ['TORCH_CUDA_ARCH_LIST']=arch
    try:
        path=_vdload(name='muse_vdense_'+kind+'_'+_vdhash.sha256(source.encode()).hexdigest()[:14],cpp_sources='',cuda_sources=source,
            functions=None,is_python_module=False,no_implicit_headers=True,
            extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr'],extra_ldflags=['-lcublasLt'],verbose=False)
    finally:
        if previous is None:_vdos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
        else:_vdos.environ['TORCH_CUDA_ARCH_LIST']=previous
    lib=_vdct.CDLL(path)
    lib.pg_algo_bytes.restype=_vdct.c_int;lib.lean_error_string.restype=_vdct.c_char_p
    lib.all_lt_options.argtypes=[_vdct.c_int]*6+[_vdct.c_size_t,_vdct.c_void_p,_vdct.c_int,_vdct.POINTER(_vdct.c_int),_vdct.POINTER(_vdct.c_int)]
    lib.all_lt_options.restype=_vdct.c_int
    lib.all_lt_run.argtypes=[_vdct.c_void_p]*5+[_vdct.c_int]*8+[_vdct.c_size_t,_vdct.c_void_p]
    lib.all_lt_run.restype=_vdct.c_int
    if kind=='lt':lib.vd_lt_info.argtypes=[_vdct.c_void_p,_vdct.c_int,_vdct.c_void_p]
    _VD_LIBS[kind]=lib;return lib


class VDLt:
    def __init__(self,w,m,fp32=False,nn=0,extended=True,workspace=None,blob=None):
        self.w=w;self.m=m;self.nn=nn;self.fp32=fp32;self.lib=vd_library('lt' if extended else 'incumbent')
        self.bank=w.T.contiguous() if nn else w
        self.out=torch.empty((m,w.shape[0]),device=w.device,dtype=torch.float32 if fp32 else torch.bfloat16)
        self.work=workspace if workspace is not None else torch.empty(1<<30,device=w.device,dtype=torch.uint8)
        self.algorithm_bytes=self.lib.pg_algo_bytes()
        if blob is not None:
            raw=bytes.fromhex(blob);assert len(raw)==self.algorithm_bytes
            self.codes=_vdct.create_string_buffer(raw,len(raw));self.count=1;self.heuristics=0
        else:
            capacity=131072;self.codes=_vdct.create_string_buffer(capacity*self.algorithm_bytes)
            count=_vdct.c_int();heur=_vdct.c_int()
            err=self.lib.all_lt_options(w.shape[0],m,w.shape[1],w.shape[0] if nn else w.shape[1],nn,int(fp32),self.work.numel(),self.codes,capacity,_vdct.byref(count),_vdct.byref(heur))
            if err:raise RuntimeError(self.lib.lean_error_string().decode())
            self.count=count.value;self.heuristics=heur.value
        self.index=0
    def __call__(self,x,index=None):
        index=self.index if index is None else index
        err=self.lib.all_lt_run(self.bank.data_ptr(),x.data_ptr(),self.out.data_ptr(),self.work.data_ptr(),self.codes,
            self.w.shape[0],self.m,self.w.shape[1],self.w.shape[0] if self.nn else self.w.shape[1],self.nn,int(self.fp32),index,0,
            self.work.numel(),torch.cuda.current_stream(x.device).cuda_stream)
        if err:raise RuntimeError(self.lib.lean_error_string().decode())
        return self.out
    def selection(self,index=None):
        index=self.index if index is None else index
        blob=bytes(self.codes)[index*self.algorithm_bytes:(index+1)*self.algorithm_bytes].hex()
        info=(_vdct.c_int*7)();self.lib.vd_lt_info(self.codes,index,info)
        return dict(family='lt',nn=self.nn,blob=blob,config=list(info))


class VDDeepGemm:
    def __init__(self,w,m,fp32=False):
        import deep_gemm
        self.mod=deep_gemm;self.w=w
        self.out=torch.empty((m,w.shape[0]),device=w.device,dtype=torch.float32 if fp32 else torch.bfloat16)
    def __call__(self,x):
        self.mod.bf16_gemm_nt(x,self.w,self.out,compiled_dims='nk');return self.out
    def selection(self):return dict(family='deep_gemm')


class VDCutlass:
    def __init__(self,w,m,fp32=False,code=0,mode=1):
        self.w=w;self.m=m;self.code=code;self.fp32=fp32;self.mode=mode;self.lib=vd_library('cutlass');self.function=vd_cubin_function(code,fp32)
        self.out=torch.empty((m,w.shape[0]),device=w.device,dtype=torch.float32 if fp32 else torch.bfloat16)
        self.work=torch.empty(0,device=w.device,dtype=torch.uint8)
        x=torch.empty((m,w.shape[1]),device=w.device,dtype=torch.bfloat16)
        needed=self._call(x,False);self.work=torch.empty(needed,device=w.device,dtype=torch.uint8)
    def _call(self,x,launch=True):
        needed=_vdct.c_size_t()
        err=self.lib.vd_cutlass(self.w.data_ptr(),x.data_ptr(),self.out.data_ptr(),self.work.data_ptr(),self.work.numel(),self.m,
            self.w.shape[0],self.w.shape[1],self.code,int(self.fp32),self.mode,
            torch.cuda.current_stream(x.device).cuda_stream,_vdct.byref(needed),int(launch),self.function)
        if err:raise RuntimeError(f'CUTLASS status={err} code={self.code}')
        return needed.value
    def __call__(self,x):
        if self.function is None:raise RuntimeError('native CUTLASS requires Hopper')
        self._call(x);return self.out
    def selection(self):return dict(family='cutlass',code=self.code,mode=self.mode)


def vd_operator(w,m,selection,fp32=False):
    family=selection['family']
    if family=='lt':return VDLt(w,m,fp32,selection['nn'],blob=selection['blob'])
    if family=='deep_gemm':return VDDeepGemm(w,m,fp32)
    if family=='cutlass':return VDCutlass(w,m,fp32,selection['code'],selection['mode'])
    raise ValueError('unknown dense family')


def vd_prepare_dispatch(state,selections):
    state['verify_dense_ops']={(m,name):vd_operator(state[name],m,selection,name=='pool_router') for (m,name),selection in selections.items()}


def vd_project(x,w,name,state,out_dtype=torch.bfloat16):
    op=state['verify_dense_ops'][x.shape[0],name]
    if op.w.data_ptr()!=w.data_ptr() or op.out.dtype!=out_dtype:raise ValueError('dense dispatch precision/weight mismatch')
    return op(x).unsqueeze(0),1
