import ctypes as vs_ct
import hashlib as vs_hash
import os as vs_os
import torch
from torch.utils.cpp_extension import load_inline as vs_load_inline

_VS_LIBS = {}


def vs_library(arch=None):
    arch = arch or ('9.0a' if torch.cuda.get_device_capability()[0] == 9 else '12.0')
    if arch not in _VS_LIBS:
        old = vs_os.environ.get('TORCH_CUDA_ARCH_LIST')
        vs_os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        try:
            path = vs_load_inline(
                name='muse_verify_stream_' + vs_hash.sha256(VS_CUDA.encode()).hexdigest()[:16] + arch.replace('.','_'),
                cpp_sources='', cuda_sources=VS_CUDA, functions=None,
                is_python_module=False, no_implicit_headers=True,
                extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','--ptxas-options=-v']
                    + (['-DVS_LOCAL'] if arch != '9.0a' else []), verbose=False)
        finally:
            if old is None:
                vs_os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                vs_os.environ['TORCH_CUDA_ARCH_LIST'] = old
        lib = vs_ct.CDLL(path)
        lib.vs_launch.argtypes = [vs_ct.c_void_p]*5 + [vs_ct.c_int]*6 + [vs_ct.c_void_p, vs_ct.c_int]
        lib.vs_launch.restype = vs_ct.c_int
        _VS_LIBS[arch] = lib
    return _VS_LIBS[arch]


class VerifyStreamingProjection:
    """Prepared BF16 weights and private split-K scratch; stable, contiguous output.

    One instance belongs to one call site. Concurrent streams require separate instances.
    Pool/router returns the 4096 pool rows + 256 hi + 256 lo as FP32, unchanged.
    """
    def __init__(self, w, m, code, splits=1, fp32=False, lib=None):
        if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.ndim != 2:
            raise ValueError('expected contiguous BF16 weight rows')
        if m < 1 or splits < 1 or code not in VS_CONFIGS:
            raise ValueError('invalid token count, split count or recipe')
        bm,bn,bk,st,pipe = VS_CONFIGS[code]
        if w.shape[1] % (bk*splits):
            raise ValueError('split does not divide the K stages')
        self.w, self.m, self.code, self.splits, self.fp32 = w,m,code,splits,fp32
        self.lib = lib or vs_library()
        self.tiles = ((w.shape[0]+bn-1)//bn)*((m+bm-1)//bm)
        self.partial = torch.empty((self.tiles,splits,bm,bn),device=w.device,dtype=torch.float32)
        self.counters = torch.zeros(self.tiles,device=w.device,dtype=torch.int32)
        self.out = torch.empty((m,w.shape[0]),device=w.device,dtype=torch.float32 if fp32 else torch.bfloat16)

    def launch(self, x, descriptor_only=False):
        if x.dtype != torch.bfloat16 or x.shape != (self.m,self.w.shape[1]) or not x.is_contiguous():
            raise ValueError('input shape, dtype or stride changed')
        e = self.lib.vs_launch(self.w.data_ptr(),x.data_ptr(),self.out.data_ptr(),
            self.partial.data_ptr(),self.counters.data_ptr(),self.m,self.w.shape[0],self.w.shape[1],
            self.code,self.splits,int(self.fp32),torch.cuda.current_stream(x.device).cuda_stream,int(descriptor_only))
        if e:
            raise RuntimeError(f'verify streaming CUDA={e} code={self.code} split={self.splits}')
        return self.out

    def __call__(self, x):
        return self.launch(x)


def vs_choices(m,n,k):
    """Compare unsplit and enough CTAs for one/two H200 waves, with no hardcoded winner."""
    for code,(bm,bn,bk,st,pipe) in VS_CONFIGS.items():
        if m <= 8 and bm != 64:
            continue
        tiles = ((n+bn-1)//bn)*((m+bm-1)//bm)
        targets = {1}
        for waves in (1,2):
            splits = 1
            while tiles*splits < 132*waves and splits < 16:
                splits *= 2
            if k % (bk*splits) == 0:
                targets.add(splits)
        for splits in sorted(targets):
            yield code,splits


def vs_prepare_dispatch(state, recipes, *, tokens=None):
    """Construct explicit receipt-selected operators; recipes maps (M,name) to (code,split)."""
    ops = {}
    for (m,name),(code,splits) in recipes.items():
        if tokens is None or m in tokens:
            ops[(m,name)] = VerifyStreamingProjection(state[name],m,code,splits,name=='pool_router')
    state['verify_streaming_ops'] = ops
    return state


def vs_project(x,w,name,state,out_dtype=torch.bfloat16):
    """Dispatch-table entry: return exactly best6's (partials, split_count) ABI."""
    if (x.shape[0],name) not in state['verify_streaming_ops']:
        raise ValueError('unprepared projection shape')
    op = state['verify_streaming_ops'][(x.shape[0],name)]
    if w.data_ptr() != op.w.data_ptr() or out_dtype != op.out.dtype:
        raise ValueError('dispatch weight or output dtype differs from prepared operator')
    return op(x).unsqueeze(0),1
