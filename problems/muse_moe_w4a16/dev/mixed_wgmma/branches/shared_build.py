"""One persistent work list over adaptive N8/N16/N32 light/heavy tiles.

PHASE uses a queue separately per GEMM. DAG additionally publishes an expert's
W2 tiles as soon as all its W13 tiles are complete, within the same launch.
No software grid barrier or residency-dependent spin condition is used.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(H),str(H.parent),str(R/'tools')]
import build
import patch_s12_permw
from build_prefill_lab import subset
from light_build import once,value

DECL='void uq_run(std::vector<int64_t>,std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> uq_info(int64_t);'

def cuda(c):
    a=c.index('template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__')
    b=c.index('// Routing only:',a)
    gemm=c[a:b]
    gemm=once(gemm,'template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__ void __launch_bounds__(128,2) gemm(',
              'template<int NT>\n__device__ __forceinline__ void uq_tile(')
    gemm=once(gemm,'int M,unsigned scale_base){','int M,unsigned scale_base,int e,int row,int mt,bool FIRST){\n constexpr int VEC=7;')
    a0=gemm.index(' int total=TASK[32]*MT;');b0=gemm.index('  int src[NJ];',a0)
    gemm=gemm[:a0]+' int count=C[e],offset=OFF[e];\n'+gemm[b0:]
    assert gemm.endswith(' }\n}\n\n');gemm=gemm[:-len(' }\n}\n\n')]+'}\n\n'
    gemm=once(gemm,'constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;',
              'constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,NJ=NT>=16?NT/16:1;const int N=FIRST?8192:4096;')
    gemm=gemm.replace('if constexpr(FIRST&&(VEC==3||VEC==5||VEC==7))','if(FIRST)').replace('else if constexpr(FIRST)','else if(FIRST)')
    # N8 writes only the eight valid rows. All threads still commit and wait.
    gemm=once(gemm,'for(int j=0;j<NJ;++j)asm volatile("cp.async.cg.shared.global',
              'for(int j=0;j<NJ;++j)if(NT>=16||t<64)asm volatile("cp.async.cg.shared.global')
    # Device functions share the same static scratch in both phases.
    gemm=once(gemm,' __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];',
              ' __shared__ float rowsum[128];\n __nv_bfloat16*epi=(__nv_bfloat16*)rowsum;')
    # All specialised tile bodies reuse exactly one dynamic/shared allocation.
    # The epi path is compile-time dead for FIRST VEC7.
    prefix=c[:c.index('template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__')]
    tail=c[b:]
    # Keep the original kernels for the S12/PERMW controls in the same module.
    original=c[c.index('template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>\n__global__'):b]
    cu=prefix+original+gemm+tail
    cu=once(cu,' int t=threadIdx.x;\n __shared__ int count[32];',
            ' int t=threadIdx.x;\n if(t<100)T8[33+t]=0;\n __shared__ int count[32];')
    cu=once(cu,'if(t<32){int n=count[t],pad=', 'if(t<32){unsigned active=__ballot_sync(~0u,count[t]>0);if(t==0)T8[35]=__popc(active);int n=count[t],pad=')
    # State is appended to the 33-entry T8 prefix allocation. Route initializes
    # all counters on every graph replay, not in prepare.
    return cu+'\n'+(H/'shared.cu.inc').read_text()


def main():
    s=(R/'problems/muse_decode_layer/cand_r3_best7_s12_final.py').read_text()
    assert hashlib.sha256(s.encode()).hexdigest().startswith('240248b2')
    perm=patch_s12_permw.patch(s);c0=value(s,'NF_CUDA');c1=value(perm,'NF_CUDA');cu=cuda(c1)
    funcs=build.strip_imports(subset(perm,('nf_plan','nf_call','nf_ilv','nf_wide','nf_perm')))
    node=next(n for n in ast.parse(perm).body if isinstance(n,ast.FunctionDef) and n.name=='nf_plan')
    plan=(ast.get_source_segment(perm,node)+'\n').replace('def nf_plan(data,base,native,streams):','def uq_plan(data,base,native,streams,dag):')
    plan=once(plan,'tasks=[torch.empty_like(off) for _ in range(3)]','tasks=[torch.empty_like(off) for _ in range(2)]+[torch.empty(133,device=dev,dtype=torch.int32)]')
    plan=once(plan,"p13,p2,pn=([t.data_ptr() for t in v] for v in (h13,h2,nv))",
              "h13[7]=tasks[0];h2[7]=tasks[0]\n    p13,p2,pn=([t.data_ptr() for t in v] for v in (h13,h2,nv))\n    pq=tasks[2].data_ptr()+33*4")
    a=plan.index('    stages=2');b=plan.index('    def route():',a)
    plan=plan[:a]+'    sms=torch.cuda.get_device_properties(x.device).multi_processor_count\n    grid=sms*4\n'+plan[b:]
    a=plan.index('    def body():');b=plan.index('    def finish():',a)
    plan=plan[:a]+'''    def body():
        _=owners;stream=torch.cuda.current_stream().cuda_stream
        if dag:mod.uq_run(p13,p2,pq,2,grid,m,stream)
        else:
            mod.uq_run(p13,p2,pq,0,grid,m,stream);mod.uq_run(p13,p2,pq,1,grid,m,stream)
'''+plan[b:]
    body=build.helpers()+'\n'+build.IMPORTS+'\n'+build.strip_imports(build.HH_CELL.read_text())+'\n'+funcs+'\n'+plan+'\n'
    body+=f'OP_CUDA={[c0,c1,cu,cu]!r}\nOP_CPP={value(s,"NF_CPP")+DECL!r}\nNF_OPTS={{}}\n'
    runtime=(H/'order_runtime.py.inc').read_text()
    runtime=once(runtime,"OP_NAMES=('S12','PERMW','ORDER','PERMW_ORDER')","OP_NAMES=('S12','PERMW','SHARED_PHASE','SHARED_DAG')")
    # Only shared modules export the new ABI; keep distinct cpp declaration sets.
    runtime=once(runtime,"for c in OP_CUDA:","for ai,c in enumerate(OP_CUDA):")
    runtime=once(runtime,"cpp_sources=OP_CPP,cuda_sources=c,","cpp_sources=OP_CPP if ai>=2 else OP_CPP.split('void uq_run(')[0],cuda_sources=c,")
    runtime=once(runtime,"'pmz_resident','pmz_run'],","'pmz_resident','pmz_run']+(['uq_run','uq_info'] if ai>=2 else []),")
    runtime=once(runtime,"mod.br_init();NF_OPTS","\n        if ai>=2:[mod.uq_info(k) for k in (0,1,2)]\n        mod.br_init();NF_OPTS")
    runtime=once(runtime,"return dict(bases=(base,perm,base,perm),native=native,","return dict(bases=(base,perm,perm,perm),native=native,")
    runtime=once(runtime,"    return nf_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm])",
                 "    if arm>=2:return uq_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm],arm==3)\n    return nf_plan(data,bank['bases'][arm],bank['native'],bank['streams'][arm])")
    runtime=runtime.replace('META exact S12 final 240248b2 vs PERMW, ORDER, PERMW_ORDER; same N32 tile math, geometry and global weight reads. ORDER changes only (expert,token-group,channel) to (expert,channel,token-group), shortening inter-CTA weight reuse distance.',
                            'META exact S12 final vs PERMW vs a shared N8/N16/N32 persistent queue across all experts. PHASE: two phases; DAG: ready expert W2 before remaining W13, one kernel. Same BF16 work, light9-16 reads once, no branch-specific grid cap. Model in whole_model.json; not a forecast of multicast rates.')
    runtime=once(runtime,"    for m in (128,256):\n        rr=", "    lines.append('RESOURCES '+str([list(OP_MODS[2].uq_info(k)) for k in (0,1,2)]))\n    for m in (128,256):\n        rr=")
    body+=runtime;compile(body,'<shared>','exec');(H/'shared_payload.py').write_text(body);(H/'cu_shared.cu').write_text(cu)
    meta=dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest())
    (H/'shared_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)


if __name__=='__main__':main()
