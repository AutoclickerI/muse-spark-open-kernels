"""Compile unchanged recoded/raw NT1 and count its isolated 16-weight decode path."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/screen/nt1_decode'


def literal(path, name):
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(name)


def extract(source):
    source = source.replace('#include <torch/extension.h>', '#include <cuda_runtime.h>')
    source = source.replace('#include <ATen/cuda/CUDAContext.h>', '')
    source = source[:source.index('size_t smem_bytes(')]
    source += '''
template __global__ void moe_fused<1>(const __nv_bfloat16*,const int*,const float*,const __nv_bfloat16*,
 const uint8_t*,const uint8_t*,const float*,const float*,__nv_bfloat16*,float*,float*,int*,__nv_bfloat16*,int,int,int);
}
'''
    return source


def probe(source, raw):
    helper = source[source.index('__device__ __forceinline__ uint32_t rotr'):source.index('struct PrefetchState')]
    # Keep exactly the production expressions, and put decode/scale/multiply on
    # separate source lines so disassembly line records distinguish the paths.
    scale = ('''unsigned s0=scale_e4m3(sc&255u);
unsigned s1=scale_e4m3((sc>>8)&255u);''' if raw else '''unsigned s0=__byte_perm(sc,0u,0x4040u)*16u+0x70007000u;
unsigned s1=__byte_perm(sc,0u,0x4141u)*16u+0x70007000u;''')
    return ('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n'
            'constexpr uint32_t DQ_MASK=0x81c081c0u,LA_MASK=0x60306030u;\n') + helper + '''
extern "C" __global__ void decode16(const uint32_t *input,uint32_t *output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 unsigned a=input[3*i],b=input[3*i+1],sc=input[3*i+2],o[8];
''' + scale + '''
 dq8(a,b,o);
 #pragma unroll
 for(int k=0;k<8;++k)o[k]=hmul2(o[k],(k&1)?s1:s0);
 ((uint4*)output)[2*i]=make_uint4(o[0],o[1],o[2],o[3]);
 ((uint4*)output)[2*i+1]=make_uint4(o[4],o[5],o[6],o[7]);
}
'''


def compile_one(name, source):
    p = OUT / (name + '.cu'); p.write_text(source)
    cube = p.with_suffix('.cubin')
    cmd = ['nvcc', '-O3', '-std=c++20', '-lineinfo', '-arch=sm_90a', '-Xptxas=-v', '-cubin', str(p), '-o', str(cube)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    p.with_suffix('.build.log').write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(r.stderr)
    r = subprocess.run(['nvdisasm', '-g', str(cube)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    p.with_suffix('.sass').write_text(r.stdout)
    counts = {}
    for op in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)', r.stdout):
        op = op.split('.')[0]; counts[op] = counts.get(op, 0) + 1
    regs = [int(v) for v in re.findall(r'Used (\d+) registers', p.with_suffix('.build.log').read_text())]
    return dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(), registers=regs, instructions=counts,
                instruction_total=sum(counts.values()))


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    recoded = literal(ROOT / 'problems/muse_moe_w4a16/cand_v9.py', 'CUDA_SRC')
    raw = literal(ROOT / 'problems/muse_decode_layer/cand_r3_best6_small_nt2.py', 'V9_CUDA_SRC')
    result = {}
    for name, src in (('recoded', recoded), ('raw', raw)):
        result[name] = compile_one(name, extract(src))
        result[name + '_decode16'] = compile_one(name + '_decode16', probe(src, name == 'raw'))
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
