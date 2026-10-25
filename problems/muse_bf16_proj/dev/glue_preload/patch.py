"""Keep immutable glue loads ahead of PDL wait in emitted Hopper code."""
import ast
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE_SHA='958f1258908f44834ec6cf0c33af00ff6c08ea62b2f5cc483a36f0a91f39d0b6'


def cuda_source(src):
    cuda=ast.literal_eval(next(n.value for n in ast.parse(src).body if isinstance(n,ast.Assign)
        and getattr(n.targets[0],'id','')=='GLUE_CL_CUDA'))
    anchor='namespace cg = cooperative_groups;'
    helpers='''
// Ordinary const/restrict loads are sunk after ACQBULK by nvcc/ptxas. Inline
// PTX memory operations preserve their place before the dependency acquire.
__device__ __forceinline__ float4 gp_load4(const float* p) {
  float4 v;
  asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3},[%4];"
      : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(p):"memory");
  return v;
}
__device__ __forceinline__ float gp_load1(const float* p) {
  float v;asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ int gp_loadi(const int* p) {
  int v;asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
'''
    assert cuda.count(anchor)==1
    cuda=cuda.replace(anchor,helpers+'\n'+anchor)
    for field in ('alpha','beta','prew','postw'):
        old=f'*reinterpret_cast<const float4*>({field} + j)'
        assert old in cuda,(field,'missing')
        cuda=cuda.replace(old,f'gp_load4({field} + j)')
    old='*reinterpret_cast<const float4*>(prew + (rank * QT + q) * 4)'
    assert old in cuda
    cuda=cuda.replace(old,'gp_load4(prew + (rank * QT + q) * 4)')
    cuda=cuda.replace('tid < EP ? bias[e] : 0.f','tid < EP ? gp_load1(bias+e) : 0.f')
    cuda=cuda.replace('const int start = *lstart;','const int start = gp_loadi(lstart);')
    return cuda


def patch(src,rows=()):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'requires S13 final 958f1258'
    source=cuda_source(src)
    suffix=(HERE/'runtime.py.inc').read_text().replace('__GP_CUDA__',repr(source),1)
    suffix=suffix.replace('__GP_ROWS__',repr(tuple(rows)),1)
    anchor='    _gate_split_prepare(state,device)\n'
    assert src.count(anchor)==1
    src=src.replace(anchor,anchor+'    gp_module()\n')
    anchor='    _best7_prepare_settings(state,weights,device)\n'
    assert src.count(anchor)==1
    src=src.replace(anchor,anchor+"    for key,plan in state['best7_plans'].items():plan['gp_key']=key\n")
    return src+'\n\n'+suffix


if __name__=='__main__':
    rows=tuple((b,q,w,role) for b,q in ((1,1),(16,1),(32,1),(1,8))
        for w in (2048,0) for role in ('b1','route','b2')) if '--trial' in sys.argv else ()
    out=patch(Path(sys.argv[1]).read_text(),rows)
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2],hashlib.sha256(out.encode()).hexdigest())
