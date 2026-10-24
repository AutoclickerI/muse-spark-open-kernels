import ctypes as _mcct
import hashlib as _mchash
import os as _mcos
from torch.utils.cpp_extension import load_inline as _mcload

_MC_LIB=None


def mc_library():
    global _MC_LIB
    if _MC_LIB is None:
        old=_mcos.environ.get('TORCH_CUDA_ARCH_LIST')
        _mcos.environ['TORCH_CUDA_ARCH_LIST']='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        try:
            path=_mcload(name='muse_memory_compression_'+_mchash.sha256(MC_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=MC_CUDA,functions=None,is_python_module=False,no_implicit_headers=True,
                extra_cuda_cflags=['-O3','-std=c++17','-lineinfo','--ptxas-options=-v'],verbose=False)
        finally:
            if old is None:_mcos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_mcos.environ['TORCH_CUDA_ARCH_LIST']=old
        _MC_LIB=_mcct.CDLL(path)
        _MC_LIB.mc_stream.argtypes=[_mcct.c_void_p]*2+[_mcct.c_size_t]+[_mcct.c_int]*2+[_mcct.c_void_p]
        _MC_LIB.mc_stream.restype=_mcct.c_int
        _MC_LIB.mc_copy_into.argtypes=[_mcct.c_void_p]*2+[_mcct.c_size_t,_mcct.c_void_p]
        _MC_LIB.mc_copy_into.restype=_mcct.c_int
    return _MC_LIB


def mc_copy_tensor(source,destination):
    if source.nbytes!=destination.nbytes or source.nbytes%16 or not source.is_contiguous() or not destination.is_contiguous():
        raise ValueError('byte-copy geometry')
    status=mc_library().mc_copy_into(source.data_ptr(),destination.data_ptr(),source.nbytes,
        torch.cuda.current_stream(source.device).cuda_stream)
    if status:raise RuntimeError('mc_copy CUDA='+str(status))


def mc_read_tensor(source,sink,ctas,depth):
    if source.nbytes%16 or sink.numel()<ctas*512:raise ValueError('stream geometry')
    status=mc_library().mc_stream(source.data_ptr(),sink.data_ptr(),source.nbytes,ctas,depth,
        torch.cuda.current_stream(source.device).cuda_stream)
    if status:raise RuntimeError('mc_stream CUDA='+str(status))
    return sink[:ctas*512]
