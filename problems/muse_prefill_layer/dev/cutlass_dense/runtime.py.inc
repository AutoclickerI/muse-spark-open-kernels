"""Source-only Hopper CUTLASS split-K/Stream-K projections. No input-value cache."""
import ctypes as _vgct
import hashlib as _vghash
import os as _vgos
import time as _vgtime
import torch

_VGC_MODULE = None
VGC_TILES = (
    (128,128,64,1,1,0,False), (128,128,64,1,2,0,False),
    (128,64,64,1,1,0,False), (128,256,64,1,1,0,False),
    (128,128,128,1,1,0,False), (128,128,64,1,1,3,False),
    (128,128,64,1,1,0,True), (128,64,64,2,1,0,True),
)

class VgcModule:
    def __init__(self, path):
        self.lib = _vgct.CDLL(path)
        p, i, z = _vgct.c_void_p, _vgct.c_int, _vgct.c_size_t
        self.lib.vgc_error.restype = _vgct.c_char_p
        self.lib.vgc_create.argtypes = [i]*11
        self.lib.vgc_create.restype = p
        self.lib.vgc_workspace.argtypes = [p]
        self.lib.vgc_workspace.restype = z
        self.lib.vgc_info.argtypes = [p,i]
        self.lib.vgc_info.restype = i
        self.lib.vgc_destroy.argtypes = [p]
        self.lib.vgc_run.argtypes = [p]*6
        self.lib.vgc_run.restype = i

    def error(self):
        return RuntimeError(self.lib.vgc_error().decode())

    def plan(self, m, n, k, dtype, recipe, device):
        return VgcPlan(self, m, n, k, dtype, recipe, device)

class VgcPlan:
    def __init__(self, module, m, n, k, dtype, recipe, device):
        # recipe: tile, decomposition(0=heuristic,1=DP,2=splitK,3=streamK),
        # splits, active SMs, raster(0=auto,1=M,2=N), swizzle, atomic reduction.
        self.module, self.handle = module, None
        tile, mode, splits, sms, raster, swizzle, atomic = recipe
        if not 0 <= tile < len(VGC_TILES) or dtype not in (torch.bfloat16,torch.float32):
            raise ValueError('invalid CUTLASS tile/dtype')
        if mode not in (0,1,2,3) or splits < 1 or (splits != 1 and mode not in (0,2)):
            raise ValueError('invalid CUTLASS split decomposition')
        if sms < 1 or raster not in (0,1,2) or swizzle not in (1,2,4,8) or atomic not in (0,1):
            raise ValueError('invalid CUTLASS scheduling recipe')
        handle = module.lib.vgc_create(tile, int(dtype==torch.float32),m,n,k,
                                       splits,mode,sms,raster,swizzle,atomic)
        if not handle:
            raise module.error()
        self.handle, self.shape, self.dtype, self.recipe = handle,(m,n,k),dtype,recipe
        try:
            size = module.lib.vgc_workspace(handle)
            self.workspace = torch.empty(max(128,size),device=device,dtype=torch.uint8)
            self.info = tuple(module.lib.vgc_info(handle,i) for i in range(3))
        except Exception:
            self.close()
            raise

    def run(self, x, w, out):
        if self.handle is None:
            raise RuntimeError('closed CUTLASS plan')
        m,n,k = self.shape
        if (x.shape!=(m,k) or w.shape!=(n,k) or out.shape!=(m,n) or
                x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or out.dtype!=self.dtype or
                not x.is_contiguous() or not w.is_contiguous() or not out.is_contiguous() or
                x.device!=w.device or x.device!=out.device or x.device!=self.workspace.device):
            raise ValueError('CUTLASS shape/layout mismatch')
        err = self.module.lib.vgc_run(self.handle,x.data_ptr(),w.data_ptr(),out.data_ptr(),
                                     self.workspace.data_ptr(),torch.cuda.current_stream(x.device).cuda_stream)
        if err:
            raise self.module.error()
        return out

    def close(self):
        if self.handle is not None:
            self.module.lib.vgc_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def vgc_module():
    global _VGC_MODULE
    if _VGC_MODULE is not None:
        return _VGC_MODULE
    if torch.cuda.get_device_capability() != (9,0):
        raise RuntimeError('CUTLASS Hopper projection requires sm_90a')
    from torch.utils.cpp_extension import load_inline
    previous = _vgos.environ.get('TORCH_CUDA_ARCH_LIST')
    _vgos.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a'
    try:
        path = load_inline(
            name='muse_verify_cutlass_'+_vghash.sha256(VGC_SOURCE.encode()).hexdigest()[:12],
            cpp_sources='',cuda_sources=VGC_SOURCE,is_python_module=False,no_implicit_headers=True,
            extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include'],
            extra_cuda_cflags=['-O3','-DNDEBUG','-std=c++17','--expt-relaxed-constexpr'],
            extra_ldflags=['-lcuda'],verbose=False)
    finally:
        if previous is None:
            _vgos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
        else:
            _vgos.environ['TORCH_CUDA_ARCH_LIST'] = previous
    _VGC_MODULE = VgcModule(path)
    return _VGC_MODULE


def vgc_choices(m,n,k,sms=132):
    # Round-robin order ensures every tile, cluster and operand family is reached
    # before tuning extra raster, grid-size and reduction settings.
    for mode,sk in ((3,1),(2,2),(2,4),(2,8),(1,1),(0,1)):
        for tile in range(len(VGC_TILES)):
            yield (tile,mode,sk,sms,0,1,0)
    for tile in range(len(VGC_TILES)):
        for mode,sk in ((3,1),(2,4)):
            for raster in (1,2):
                yield (tile,mode,sk,sms,raster,1,0)
            for use_sms in (max(8,sms//2//2*2),max(8,(sms//4*3)//2*2)):
                yield (tile,mode,sk,use_sms,0,1,0)
            yield (tile,mode,sk,sms,0,1,1)
        yield (tile,2,16,sms,0,1,0)
        yield (tile,3,1,sms,2,4,0)
