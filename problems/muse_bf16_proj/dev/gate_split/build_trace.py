"""Instrument the frozen gate-split chain with per-CTA globaltimer probes."""
import ast
import hashlib
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
SRC=REPO/'prof/lab/gate_split.py'
PROBE=r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
__device__ unsigned long long* gs_probe_ptr=nullptr;
__device__ __forceinline__ unsigned long long gs_time() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)); return t;
}
struct GsScope {
  unsigned long long* p;
  __device__ GsScope(int phase) {
    int c=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
    p=gs_probe_ptr && c<65536 ? gs_probe_ptr+(phase*65536+c)*2 : nullptr;
    if(p && threadIdx.x==0)p[0]=gs_time();
  }
  __device__ ~GsScope(){if(p && threadIdx.x==0)p[1]=gs_time();}
};
void gs_set_trace(int64_t pointer) {
  auto p=reinterpret_cast<unsigned long long*>(pointer);
  auto e=cudaMemcpyToSymbol(gs_probe_ptr,&p,sizeof(p));
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
'''


def traced_cuda(src):
    for phase,name in enumerate(('attn_main','attn_combine')):
        pattern=r'(__global__[^{}]*\b'+name+r'\([^{}]*\)\s*\{)'
        src,n=re.subn(pattern,lambda m:m[1]+f'\n  GsScope gs_probe({phase});',src)
        assert n==1 or name=='attn_combine',name
    return PROBE+src


def main():
    src=SRC.read_text();lines=src.splitlines();replacements=[]
    for n in ast.parse(src).body:
        if not isinstance(n,ast.Assign):continue
        name=getattr(n.targets[0],'id',None)
        if name in ('RO_PORT_CUDA_GS','RO_LAYER_CUDA_GS'):
            replacements.append((n.lineno,n.end_lineno,f'{name} = {traced_cuda(ast.literal_eval(n.value))!r}'))
        elif name in ('RO_PORT_CPP','RO_LAYER_CPP'):
            replacements.append((n.lineno,n.end_lineno,f'{name} = {ast.literal_eval(n.value)+chr(10)+"void gs_set_trace(int64_t);"!r}'))
    assert len(replacements)==4
    for start,end,value in sorted(replacements,reverse=True):lines[start-1:end]=[value]
    src='\n'.join(lines)+'\n'
    anchor='        tag = hashlib.sha256((cpp + cuda).encode()).hexdigest()[:10]'
    assert src.count(anchor)==1
    src=src.replace(anchor,"        if kind.endswith('_gs'): fns=fns+['gs_set_trace']\n"+anchor)
    tree=ast.parse(src);lines=src.splitlines()
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gate_split_kernel')
    first=min([d.lineno for d in node.decorator_list]+[node.lineno])
    kern='\n'.join(lines[first-1:node.end_lineno]).replace('_gate_split_kernel(','_gt_kernel(')
    kern=kern.replace('CNT, Y, G, T,','CNT, Y, G, STAMP, T,')
    kern=kern.replace('    ps = tl.program_id(1)','    ps = tl.program_id(1)\n    sp = STAMP + (ps * (N // BN) + pn) * 8\n    tl.store(sp, _gt_time())')
    # Observe the accumulator only after its stores: a timestamp directly
    # after tl.dot could be scheduled before an asynchronous MMA drains.
    kern=kern.replace('    tl.debug_barrier()', '    tl.debug_barrier()\n    tl.store(sp + 1, _gt_time())',1)
    kern=kern.replace("    arrived = tl.atomic_add",'    tl.store(sp + 2, _gt_time())\n    arrived = tl.atomic_add')
    kern=kern.replace('    if arrived == S - 1:',
        '    tl.store(sp + 3, arrived.to(tl.int64))\n    tl.store(sp + 4, 0)\n    tl.store(sp + 5, 0)\n    tl.store(sp + 6, 0)\n    if arrived == S - 1:\n        tl.store(sp + 4, _gt_time())')
    kern=kern.replace('        tl.extra.cuda.gdc_wait()','        tl.extra.cuda.gdc_wait()\n        tl.store(sp + 5, _gt_time())')
    kern+='\n        tl.store(sp + 6, _gt_time())\n'
    timer='''
@triton.jit
def _gt_time():
    return tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l', [], dtype=tl.int64, is_pure=False, pack=1)
'''
    out=src+'\n'+timer+'\n'+kern+'\n\n'+(HERE/'trace_body.py').read_text()
    compile(out,'gate_trace','exec')
    path=HERE/'lab_gate_trace.py';path.write_text(out)
    print(path,hashlib.sha256(out.encode()).hexdigest())


if __name__=='__main__':main()
