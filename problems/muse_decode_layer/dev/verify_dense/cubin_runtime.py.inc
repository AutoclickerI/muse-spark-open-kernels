_VD_CUBIN_MOD=None
_VD_CUBIN_FUNCS={}


def vd_cubin_function(code,fp32):
    global _VD_CUBIN_MOD
    if torch.cuda.get_device_capability()[0]!=9:
        return None  # Local descriptor tests use no device function; callers force portable math.
    import base64,zlib
    driver=_vdct.CDLL('libcuda.so.1')
    driver.cuModuleLoadData.argtypes=[_vdct.POINTER(_vdct.c_void_p),_vdct.c_void_p];driver.cuModuleLoadData.restype=_vdct.c_int
    driver.cuModuleGetFunction.argtypes=[_vdct.POINTER(_vdct.c_void_p),_vdct.c_void_p,_vdct.c_char_p];driver.cuModuleGetFunction.restype=_vdct.c_int
    if _VD_CUBIN_MOD is None:
        data=_vdct.create_string_buffer(zlib.decompress(base64.b64decode(VD_CUBIN)));mod=_vdct.c_void_p()
        err=driver.cuModuleLoadData(_vdct.byref(mod),data)
        if err:raise RuntimeError(f'CUTLASS module load CUDA={err}')
        _VD_CUBIN_MOD=mod
    key=f'{code}:{int(fp32)}'
    if key not in _VD_CUBIN_FUNCS:
        func=_vdct.c_void_p();err=driver.cuModuleGetFunction(_vdct.byref(func),_VD_CUBIN_MOD,VD_CUBIN_FUNCTIONS[key].encode())
        if err:raise RuntimeError(f'CUTLASS function CUDA={err}')
        _VD_CUBIN_FUNCS[key]=func
    return _VD_CUBIN_FUNCS[key]
