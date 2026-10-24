"""Rebuild the actual sm90a scalar/plane/vector cubins, dump SASS and audit K loops.

Run inside tools/img. No Hopper GPU or device execution is needed.
"""
from pathlib import Path
import subprocess
from engine import mw_compile
from sass_audit import audit
import json

HERE=Path(__file__).resolve().parent
OUT=HERE/'diagnostics'
INCLUDE=Path('/opt/sglang/lib/python3.12/site-packages/tilelang')


def build(name,source):
    cu=OUT/(name+'.cu');cubin=OUT/(name+'.cubin');cu.write_text(source)
    p=subprocess.run(['nvcc','--cubin','-O3','-lineinfo','-arch=sm_90a','-std=c++20','-Xptxas=-v',
                      '-I'+str(INCLUDE/'src'),'-I'+str(INCLUDE/'3rdparty/cutlass/include'),
                      str(cu),'-o',str(cubin)],capture_output=True,text=True)
    (OUT/(name+'.ptxas.log')).write_text(p.stdout+p.stderr)
    p.check_returncode()
    sass=subprocess.run(['cuobjdump','--dump-sass',str(cubin)],capture_output=True,text=True,check=True)
    (OUT/(name+'.sass')).write_text(sass.stdout)


def main():
    OUT.mkdir(exist_ok=True)
    for name,planes in (('scalar',False),('plane',True)):
        build(name,mw_compile(128,8192,BN=32,GRID=264,target='sm_90a',PLANES=planes).get_kernel_source())
    source=(HERE/'vector.cu').read_text()
    source='#include <cuda_bf16.h>\n#include <stdint.h>\n'+source[source.index('namespace mwvec'):source.index('\nvoid vec_run')]
    for nt in (16,32,64):
        inst=f'\ntemplate __global__ void mwvec::grouped<{nt}>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,float*,int,int,int);\n'
        build(f'vector{nt}',source+inst)
    results={name:audit(OUT/(name+'.sass')) for name in ('scalar','plane')}
    results.update({f'vector{nt}':audit(OUT/f'vector{nt}.sass',64) for nt in (16,32,64)})
    (OUT/'sass_counts.json').write_text(json.dumps(results,indent=2)+'\n')
    for name,v in results.items():print(name,v['static_loop_instructions'],v['instructions_per_packed_byte'],v['local_load_store'],flush=True)


if __name__=='__main__':main()
