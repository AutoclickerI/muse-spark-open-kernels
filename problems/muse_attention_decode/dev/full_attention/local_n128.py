"""Local-only N128 launch: retain arithmetic, put transposed V in global scratch.

SM120's 99KiB shared limit cannot hold the Hopper ring. K stays shared; exact
FP8 V bytes are gathered directly into their permuted PV operand in scratch.
No candidate or submitted payload imports this module.
"""
import hashlib
from types import SimpleNamespace


def source(cuda):
    def edit(a,b):
        nonlocal cuda
        assert cuda.count(a)==1,(a[:80],cuda.count(a))
        cuda=cuda.replace(a,b)
    # Keep aggregate initialization order intact: append pointer after sc.
    edit('k_scale is applied on device\n};','k_scale is applied on device\n  uint8_t* local_vt;\n};')
    # S15's never-used non-Hopper N128 branch indexes WG1's Q beyond its head.
    # Native WGMMA supplies warp-local rows implicitly; make the stand-in agree.
    edit('w=threadIdx.x>>5;', 'w=(threadIdx.x>>5)&3;')
    edit('  uint8_t* VR = Ks + ST * 16384;\n  uint8_t* VT = VR + ST * 16384;',
         '  uint8_t* VT = p.local_vt + (int64_t)blockIdx.x * ST * 16384;\n  uint8_t* VR = VT;')
    edit('      transpose_group(VR + slot * 16384, VT + slot * 16384, pw, lane);',
         '      __threadfence_block(); __syncwarp(); // exact global V stores precede publication')
    edit('        cp16(vd + off(kk, cb), p.vc + src, valid);', '''        const int slab=kk>>6, row64=kk&63, k16=row64&15;
        const int perm=(row64&~15)+4*((k16&7)>>1)+(k16&1)+2*(k16>>3);
        uint8_t* vt=VT+slot*16384+head*8192+slab*4096;
        #pragma unroll
        for(int d=0;d<16;d++)vt[off(cb+d,perm)]=valid?p.vc[src+d]:0;''')
    edit('return CLKPAD + 24576 + ST * 49152;', 'return CLKPAD + 24576 + ST * 16384;')
    edit('int64_t split_stride, int64_t flags, int64_t stream) {',
         'int64_t split_stride, int64_t flags, int64_t stream, int64_t local_vt) {')
    edit('  // A segment has at most grid/NS', '  p.local_vt=(uint8_t*)local_vt;\n  // A segment has at most grid/NS')
    return cuda


def install(m):
    cuda=source(m.VQ8_CUDA)
    cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n'+m.VQ8_CPP
    # Pointer-only pybind declaration in S15 has unnamed arguments.
    at=cpp.index('void vq8_forward(');end=cpp.index(');',at)
    cpp=cpp[:end]+', int64_t'+cpp[end:]
    cache=[]
    def mod():
        if not cache:
            arch=''.join(map(str,m.torch.cuda.get_device_capability()))
            raw=m.load_inline(name='p5_n128_local_'+hashlib.sha256(cuda.encode()).hexdigest()[:12]+'_'+arch,
                cpp_sources=cpp,cuda_sources=cuda,functions=['vq8_forward','vq8_resources'],
                no_implicit_headers=True,with_pytorch_error_handling=False,
                extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
            buffers={}
            def forward(*args):
                grid,stages=args[21],args[22]
                key=(grid,stages)
                if key not in buffers:buffers[key]=m.torch.empty((grid,stages,16384),device='cuda',dtype=m.torch.uint8)
                return raw.vq8_forward(*args,buffers[key].data_ptr())
            cache.append(SimpleNamespace(vq8_forward=forward,vq8_resources=raw.vq8_resources))
        return cache[0]
    m._vq8_module=mod
    mod()
    return dict(local_vq8='N128 original arithmetic/ring; exact V gather+permutation in global scratch')
