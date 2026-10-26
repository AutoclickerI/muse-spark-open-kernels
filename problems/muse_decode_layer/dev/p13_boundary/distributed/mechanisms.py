"""Distributed P13 mechanisms on the pinned newest P4 production source."""
import ast
import base64
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path
H=Path(__file__).resolve().parent
R=H.parents[4]
P=R/'problems/muse_decode_layer'
O=H/'out';O.mkdir(exist_ok=True)

def sha(s):return hashlib.sha256(s.encode()).hexdigest()

def once(s,a,b):
    assert s.count(a)==1,(a[:80],s.count(a));return s.replace(a,b,1)

def base():
    lines=(R/'docs/PLAN.md').read_text().splitlines()
    row=next(line for line in lines if re.match(r'^\|\s*\d+\s*\|\s*P4\s*\|',line))
    published='\n'.join([row]+[line for line in lines if line.startswith('- P4 S')])
    versions=[(int(n),prefix) for n,prefix in re.findall(r'\bS(\d+)\s+`?([0-9a-f]{8})',published) if int(n)>=21]
    assert versions,'No published S2x source identity in P4 PLAN row'
    version=max(n for n,_ in versions);prefix=next(p for n,p in versions if n==version)
    manifest=json.loads((P/f'dev/s{version}/manifest.json').read_text())
    assert manifest['sha256'].startswith(prefix),'P4 source/PLAN publication in progress; rebuild after its identity agrees'
    packed=(R/manifest['shipped_source']).read_text()
    assert sha(packed)==manifest['shipped_sha256']
    node=next(n for n in ast.parse(packed).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_source' for t in n.targets))
    blob=next(n.value for n in ast.walk(node.value) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
    s=lzma.decompress(base64.b85decode(blob)).decode();assert sha(s)==manifest['sha256']
    return s,dict(manifest,p13_base_series='S'+str(version))

def literal(s,name):
    n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets))
    return ast.literal_eval(n.value)

def function(s,name):
    n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    prefix='@triton.jit\n' if n.decorator_list else ''
    return prefix+ast.get_source_segment(s,n)

def boundary_suffix(source):
    old=literal(source,'GLUE_CL_CUDA')
    # Retain the exact S21 cluster route arithmetic and top8, changing only
    # the row scale before pooled norm / sigmoid. No metadata producer here.
    a=old.index('template <int C, int NT>\n__global__ void __launch_bounds__(NT) route_cl')
    b=old.index('  if (meta != nullptr && rank == 0)',a)
    route=old[a:b]+'}\n'
    route=once(route,'route_cl(','scaled_route(')
    route=once(route,'MoERouteMeta* __restrict__ meta, unsigned* __restrict__ arrive)',
               'const float* __restrict__ stats, int parts)')
    route=once(route,'  float4 p[GQ];','  float sum_stats=0.f;for(int c=0;c<parts;++c)sum_stats+=__ldcg(stats+t*parts+c);\n  const float row_scale=rsqrtf(sum_stats/8192.f+1.e-5f);\n  float4 p[GQ];')
    mark='    v[0] += p[g].x * p[g].x'
    assert route.count(mark)==1
    route=route.replace(mark,'    p[g].x*=row_scale;p[g].y*=row_scale;p[g].z*=row_scale;p[g].w*=row_scale;\n'+mark)
    route=once(route,'__expf(-logit * 0.1767766952966369f)','__expf(-(logit*row_scale) * 0.1767766952966369f)')
    helpers=old[:old.index('constexpr int HID = 8192;')]
    helpers+='constexpr int DP=4096,NE=256;\n'
    a=old.index('__device__ __forceinline__ uint32_t route_ord(');b=old.index('// grid (C, T)',a)
    helpers+=old[a:b]
    cuda=(H/'boundary.cu').read_text()+'\n'+helpers+route+'''
template<int C,int NT>void p13_scaled_launch(std::vector<int64_t>a,int t,int parts,cudaStream_t stream) {
 auto fn=glue_cl::scaled_route<C,NT>;cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(C,t);cfg.blockDim=dim3(NT);cfg.stream=stream;
 cudaLaunchAttribute attr[2]{};attr[0].id=cudaLaunchAttributeClusterDimension;
 attr[0].val.clusterDim.x=C;attr[0].val.clusterDim.y=attr[0].val.clusterDim.z=1;
 attr[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;attr[1].val.programmaticStreamSerializationAllowed=1;
 cfg.attrs=attr;cfg.numAttrs=2;
 auto rc=cudaLaunchKernelEx(&cfg,fn,(const void*)a[0],(int)a[1],(int)a[2],(long long)a[3],(long long)a[4],
  (const float*)a[5],(const float*)a[6],(const int*)a[7],(__nv_bfloat16*)a[8],(int*)a[9],(float*)a[10],(int*)a[11],(const float*)a[12],parts);
 if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
}
void p13_scaled_route(std::vector<int64_t>a,int64_t t,int64_t c,int64_t nt,int64_t parts,int64_t stream) {
 if(a.size()!=13)throw std::runtime_error("P13 scaled route ABI");
 if(c==8&&nt==256)return p13_scaled_launch<8,256>(a,t,parts,(cudaStream_t)stream);
 if(c==1&&nt==256)return p13_scaled_launch<1,256>(a,t,parts,(cudaStream_t)stream);
 throw std::runtime_error("P13 scaled route geometry");
}
'''
    # helpers opens glue_cl; close it before global launch bindings.
    cuda=once(cuda,'template<int C,int NT>void p13_scaled_launch','} // namespace glue_cl\ntemplate<int C,int NT>void p13_scaled_launch')
    cpp='#include <cstdint>\n#include <vector>\n#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\nvoid p13_b1(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);\nvoid p13_scaled_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);'
    triton=function(source,'_route_fast').replace('def _route_fast(','def _p13b_route_fast(')
    triton=once(triton,'X,IDS,WTS,LIDS,REPLICAS:tl.constexpr','X,IDS,WTS,LIDS,STATS,NPARTS:tl.constexpr,REPLICAS:tl.constexpr')
    triton=once(triton,'    norm=p*tl.rsqrt','    row_scale=tl.rsqrt(tl.sum(tl.load(STATS+t*NPARTS+tl.arange(0,NPARTS)),0)/8192+1.e-5)\n    p*=row_scale;logit*=row_scale\n    norm=p*tl.rsqrt')
    return 'P13B_CUDA='+repr(cuda)+'\nP13B_CPP='+repr(cpp)+'\n'+triton+'\n'+(H/'boundary.py.inc').read_text(),cuda

def route_suffix(source):
    vendor=literal(source,'NF_VENDOR')
    blob=next(n.value for n in ast.walk(ast.parse(vendor)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>1000)
    import zlib
    vendor=zlib.decompress(base64.b64decode(blob)).decode()
    template=literal(vendor,'V18S_CUDA_TEMPLATE')
    ns={'re':re,'V18S_CUDA_TEMPLATE':template}
    exec(function(source,'_port_cuda'),ns);ns['V18S_CUDA_TEMPLATE']=ns['_port_cuda'](template)
    exec(function(vendor,'v18s_cuda_source'),ns)
    cuda=ns['v18s_cuda_source']([(1,1,1,1,0,0,256,1,4608,8192,1,8,16,1,6)])
    # Keep reviewed own S19 top8/norm code; use exactly one row per last CTA.
    common=(H.parent/'fused.cu.inc').read_text();common=common[:common.index('// @@METADATA_DEVICE@@')]
    common=once(common,'int* lids; unsigned* arrive;','int* lids; unsigned* arrive; int rows;')
    a=common.index('__device__ __forceinline__ void p13_small_finish(')
    common=common[:a]+'''__device__ __forceinline__ void p13_row_finish(const float* p,int n,P13SmallCtx ctx) {
  if(p13_last(ctx.arrive,gridDim.x/ctx.rows))p13_route_row(p,1,1,n,n,0,ctx);
}
'''
    cuda=common+cuda
    cuda=once(cuda,'    uint32_t U0, uint32_t U1) {\n  const int m = CM',
              '    uint32_t U0, uint32_t U1, P13SmallCtx ctx) {\n  const int m = CM')
    signature='int S, int64_t t01, int64_t u01, int64_t stream, int query) {'
    assert cuda.count(signature)==2;cuda=cuda.replace(signature,signature.replace('int query) {','int query, P13SmallCtx ctx) {'))
    cuda=once(cuda,'stream, query);','stream, query, ctx);')
    cuda=once(cuda,'(uint32_t)((uint64_t)u01 >> 32));','(uint32_t)((uint64_t)u01 >> 32),ctx);')
    a=cuda.index('__global__ void __launch_bounds__(NTMAX) v18s_gemv(')
    b=cuda.index('\nstatic int v18s_smem_optin()',a)
    body=cuda[a:b].replace('blockIdx.x','p13_tile').replace('gridDim.x','(gridDim.x/ctx.rows)')
    prefix='''  const int p13_row=blockIdx.x%ctx.rows,p13_tile=blockIdx.x/ctx.rows;
  x+=(long long)p13_row*K_arg;outp=static_cast<float*>(outp)+(long long)p13_row*N_arg;
  ctx.x+=(long long)p13_row*4096;ctx.ids+=p13_row*8;ctx.wts+=p13_row*8;ctx.lids+=p13_row*8;ctx.arrive+=p13_row;
'''
    body=once(body,'  const int m = CM',prefix+'  const int m = CM')
    marker='\n    }\n  }\n}\n'
    assert body.endswith(marker)
    body=body[:-len(marker)]+'\n    }\n  }\n  p13_row_finish(static_cast<const float*>(outp),N,ctx);\n}\n'
    cuda=cuda[:a]+body+cuda[b:]
    cuda=once(cuda,'lc.gridDim = dim3(N / RB, S, 1);','lc.gridDim = dim3((N / RB)*ctx.rows, S, 1);')
    cuda+='''
int p13r_launch(int inst,int64_t wp,int64_t x,int64_t out,int64_t esc,int m,int N,int K,int WR,int WK,int KU,int S,int64_t t01,int64_t u01,int64_t stream,int query,std::vector<int64_t> v) {
 if(v.size()!=9)throw std::runtime_error("P13 row pointer ABI");
 P13SmallCtx ctx{(const float*)v[0],(const float*)v[1],(const int*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(float*)v[5],(int*)v[6],(unsigned*)v[7],(int)v[8]};
 return v18s_launch(inst,wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query,ctx);
}
'''
    cpp='#include <cstdint>\n#include <vector>\n#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\nint p13r_launch(int,int64_t,int64_t,int64_t,int64_t,int,int,int,int,int,int,int,int64_t,int64_t,int64_t,int,std::vector<int64_t>);'
    return 'P13R_CUDA='+repr(cuda)+'\nP13R_CPP='+repr(cpp)+'\n'+(H/'route.py.inc').read_text(),cuda

def main():
    kind='route' if '--route' in sys.argv else 'boundary'
    s,meta=base();suffix,cuda=route_suffix(s) if kind=='route' else boundary_suffix(s)
    (O/'base.py').write_text(s);(O/(kind+'.cu')).write_text(cuda)
    sys.path.insert(0,str(P/'dev/s16'));from pack import pack
    arms={'A':suffix,'B':suffix.replace('P13R_ROWS=(1,8,16)','P13R_ROWS=(1,)') if kind=='route' else suffix.replace('P13B_WIDE=4','P13B_WIDE=8')}
    for name,tail in arms.items():
        full=s+'\n'+tail;compile(full,'<P13B-'+name+'>','exec')
        (O/(kind+'_'+name+'.py')).write_text(full);(O/(kind+'_'+name+'_packed.py')).write_text(pack(full))
        (O/(kind+'_'+name+'.inc')).write_text(tail)
    report=dict(base_sha256=sha(s),base_shipped_sha256=meta['shipped_sha256'],base_series=meta['p13_base_series'],
                arms={n:dict(sha256=sha(s+'\n'+tail),suffix_sha256=sha(tail)) for n,tail in arms.items()},
                cuda_sha256=sha(cuda),mechanism='Per-row independent M1 pool/router grid, tile-major interleaving for weight reuse, per-row last arrival computes exactly one norm/top8 row; no cross-row serialization.' if kind=='route' else 'Distributed b1 residual + unnormalized weighted BF16 activation; per-CTA FP32 row moments, RMS scalar deferred through unchanged pool/router GEMM into route. No changed weight bytes.',
                changed_rounding='Unchanged BF16 input and weights; independent M1 GEMM accumulation order.' if kind=='route' else 'BF16(st*g) before GEMM, FP32 scalar after GEMM, instead of BF16(st*rms*g); TASK decides tolerance.',
                restrictions='FP32 residual and all weight bits unchanged; each route epilogue owns one row; M1/B subset is a prespecified arm.' if kind=='route' else 'FP32 residual and all weights unchanged; v1F coupled RT path unchanged; rows independent CTAs/clusters, no serial row loop.')
    (H/(kind+'_manifest.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
