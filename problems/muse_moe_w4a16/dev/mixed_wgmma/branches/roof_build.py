"""Freeze S12's nf cell and derive three attributable diagnostic kernels.

Only diagnostics: trivial/loads arms are intentionally incorrect and never
replace the lab host's returned computation. The normal source is unchanged.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[4]
sys.path[:0]=[str(H),str(H.parent),str(R/'tools')]
from build_prefill_lab import subset
import build


def once(s,a,b):
    assert s.count(a)==1,a[:100]
    return s.replace(a,b)


def body_span(s,anchor):
    a=s.index(anchor);lo=s.index('{',a);depth=1;i=lo+1
    while depth:
        if s[i]=='{':depth+=1
        if s[i]=='}':depth-=1
        i+=1
    return lo+1,i-1


def probes(c,loads=False):
    c=once(c,'namespace mwbr {','''namespace mwbr {
__device__ __forceinline__ unsigned nd_hash(unsigned a,unsigned b,unsigned s){
 unsigned v=a^b^s;return (v&0x007f007fu)|0x3f803f80u;
}
''')
    a,b=body_span(c,'void dq(')
    c=c[:a]+'''
 unsigned v=nd_hash(a,b,raw);
 #pragma unroll
 for(int i=0;i<8;++i)out[i]=v;
'''+c[b:]
    # Both q words and all scale bits remain inputs to an actual instruction.
    a=c.index('          const uint32_t Sr =')
    b=c.index('            for (int j = 0; j < NT; ++j) mma16816',a)
    c=c[:a]+'''          unsigned v=wv.x^wv.y^wv.z^wv.w^sw;
          v=(v&0x007f007fu)|0x3f803f80u;
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const unsigned a0=v,a1=v,a2=v,a3=v;
#pragma unroll
'''+c[b:]
    if not loads:return c
    # Keep the producer/copy ring and task traversal; remove MMAs and epilogues.
    a,b=body_span(c,'void mma_group(')
    c=c[:a]+'''
 unsigned v=a[0]^a[8];c[0]+=float(v&255u);
'''+c[b:]
    # WGMMA operand waits/fences have no consumer in this arm. cp.async's waits,
    # CTA barriers and proxy fence stay, as part of the original load schedule.
    lines=c.splitlines(keepends=True)
    c=''.join(l for l in lines if not ('asm volatile("wgmma.' in l and '.mma_async.' not in l))
    a=c.index('  if constexpr(FIRST&&')
    b=c.index('\n  __syncthreads();\n }\n}',a)
    c=c[:a]+'''  if(t==0)Y[blockIdx.x]=acc[0];
'''+c[b:]
    # Preserve every cached activation load even after removal of its MMA use.
    marker='namespace pmz {'
    c=once(c,marker,marker+'''
template<bool CG>__device__ __forceinline__ uint4 nd_ld(const uint4*p){
 uint4 v;
 if constexpr(CG)asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
  :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");
 else asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4];"
  :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");return v;
}
''')
    c=c.replace('v0 = __ldcg(p); v1 = __ldcg(p + 1);','v0 = nd_ld<true>(p); v1 = nd_ld<true>(p + 1);')
    c=c.replace('v0 = __ldg(p); v1 = __ldg(p + 1);','v0 = nd_ld<false>(p); v1 = nd_ld<false>(p + 1);')
    c=once(c,'  uint32_t Bf[NT][8][8];','  uint32_t Bf[NT][8][8];unsigned nd_sum=0;')
    a=c.index('      float acc[NT][4];')
    b=c.index('      __syncwarp();',a)
    c=c[:a]+'''      const uint4* wq=reinterpret_cast<const uint4*>(tile)+warp*8*32+lane;
      const uint4 scv=*(reinterpret_cast<const uint4*>(tile+TILE_W_BYTES)+warp*32+lane);
      nd_sum^=scv.x^scv.y^scv.z^scv.w;
#pragma unroll
      for(int g=0;g<8;++g){uint4 v=wq[g*32];nd_sum^=v.x^v.y^v.z^v.w;}
'''+c[b:]
    a=c.index('      float* rp = red +')
    b=c.index('      buf ^= 1;',a)
    c=c[:a]+'''      bar_consumers();
      if(tid==0)prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);
'''+c[b:]
    c=once(c,'\n}\n\nsize_t smem_bytes(int nt, int nstage, int M)',
           '\n y[blockIdx.x*NTHR+tid]=__uint_as_float((nd_sum&0x007fffffu)|0x3f800000u);\n}\n\nsize_t smem_bytes(int nt, int nstage, int M)')
    return c


INFO=r'''
std::vector<int64_t> nd_info(int64_t first,int64_t M,int64_t stages){
 cudaFuncAttributes a{};int n;
 if(first){
  BR_CUDA_CHECK(cudaFuncGetAttributes(&a,pmz::phase<1,true,true,true,16>));
  BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,true,true,true,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));
 }else{
  BR_CUDA_CHECK(cudaFuncGetAttributes(&a,pmz::phase<1,false,true,false,16>));
  BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,false,true,false,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));
 }
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)pmz::smem_bytes(1,stages,M),n};
}
'''


def main():
    src=R/'problems/muse_decode_layer/cand_r3_best7_s12_final.py'
    if not src.exists():src=R/'problems/muse_decode_layer/cand_r3_best7_s12.py'
    text=src.read_text();tree=ast.parse(text)
    cu=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NF_CUDA')
    cpp=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NF_CPP')
    names=['nf_plan','nf_call','nf_ilv']
    if 'def nf_wide(' in text:names.append('nf_wide')
    functions=build.strip_imports(subset(text,tuple(names)))
    hh=build.strip_imports(build.HH_CELL.read_text())
    # Shared helper bank transforms, not the old branch policy or tuners.
    plan=(H/'plan.py.inc').read_text()
    transforms=subset(plan,('br_ilv_rows','br_ilv_bank','br_wide'))
    cudas=[cu+INFO,probes(cu)+INFO,probes(cu,True)+INFO]
    out=H/'roof_payload.py'
    body=build.helpers()+'\n'+build.IMPORTS+'\n'+hh+'\n'+transforms+'\n'+functions+'\n'
    body+=f'ND_CUDA={cudas!r}\nND_CPP={cpp+"std::vector<int64_t> nd_info(int64_t,int64_t,int64_t);"!r}\nNF_OPTS={{}}\nND_WIDE={"F(26,32,true,7,0,0,0)" in cu!r}\n'
    body+=f'ND_STREAM_CUDA={("#include <stdexcept>\n"+(H/"roof_stream.cu.inc").read_text())!r}\n'
    body+=(H/'roof_runtime.py.inc').read_text()
    compile(body,str(out),'exec');out.write_text(body)
    manifest=dict(source=str(src.relative_to(R)),source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  nf_cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),payload_sha256=hashlib.sha256(body.encode()).hexdigest(),
                  cuda_sha256=[hashlib.sha256(c.encode()).hexdigest() for c in cudas])
    (H/'roof_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for i,c in enumerate(cudas):(H/f'cu_roof{i}.cu').write_text(c)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
