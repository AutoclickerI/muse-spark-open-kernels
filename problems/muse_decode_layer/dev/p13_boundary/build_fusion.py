"""Exact S19 patch: small-M pool epilogue plus large-M route/metadata fusion."""
import ast
import base64
import hashlib
import json
import lzma
import re
import sys
import zlib
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path.insert(0,str(H))
from p13_build import base,sha


def once(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a))
    return s.replace(a,b,1)


def sources():
    s=base();tree=ast.parse(s)
    def literal(name):
        n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets))
        return ast.literal_eval(n.value)
    vendor=literal('NF_VENDOR')
    blob=next(n.value for n in ast.walk(ast.parse(vendor)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>1000)
    vendor=zlib.decompress(base64.b64decode(blob)).decode()
    ns={'re':re}
    node=next(n for n in ast.parse(vendor).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='V18S_CUDA_TEMPLATE' for t in n.targets))
    ns['V18S_CUDA_TEMPLATE']=ast.literal_eval(node.value)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_port_cuda')
    exec(ast.get_source_segment(s,node),ns)
    ns['V18S_CUDA_TEMPLATE']=ns['_port_cuda'](ns['V18S_CUDA_TEMPLATE'])
    node=next(n for n in ast.parse(vendor).body if isinstance(n,ast.FunctionDef) and n.name=='v18s_cuda_source')
    exec(ast.get_source_segment(vendor,node),ns)
    insts=[(1,mt,1,1,0,0,256,0,4608,8192,1,8,16,1,6) for mt in (1,2)]
    pool=ns['v18s_cuda_source'](insts)
    common=(H/'fused.cu.inc').read_text()
    pool_common=common[:common.index('// @@METADATA_DEVICE@@')]
    pool=pool_common+pool
    pool=once(pool,'    uint32_t U0, uint32_t U1) {\n  const int m = CM', '    uint32_t U0, uint32_t U1, P13SmallCtx ctx) {\n  const int m = CM')
    signature='int S, int64_t t01, int64_t u01, int64_t stream, int query) {'
    assert pool.count(signature)==2
    pool=pool.replace(signature,signature.replace('int query) {','int query, P13SmallCtx ctx) {'))
    assert pool.count('stream, query);')==2
    pool=pool.replace('stream, query);','stream, query, ctx);')
    pool=once(pool,'(uint32_t)((uint64_t)u01 >> 32));','(uint32_t)((uint64_t)u01 >> 32), ctx);')
    marker='\n    }\n  }\n}\n\nstatic int v18s_smem_optin()'
    pool=once(pool,marker,'\n    }\n  }\n  p13_small_finish(static_cast<const float*>(outp),m,N,ctx);\n}\n\nstatic int v18s_smem_optin()')
    pool+='''
int p13_pool_launch(int inst,int64_t wp,int64_t x,int64_t out,int64_t esc,int m,int N,int K,int WR,int WK,int KU,int S,int64_t t01,int64_t u01,int64_t stream,int query,std::vector<int64_t> v) {
 if(v.size()!=8)throw std::runtime_error("P13 small pointer ABI");
 P13SmallCtx ctx{(const float*)v[0],(const float*)v[1],(const int*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(float*)v[5],(int*)v[6],(unsigned*)v[7]};
 return v18s_launch(inst,wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query,ctx);
}
'''
    cu=literal('P3B_CUDA');a=cu.index('template<bool META>__global__');b=cu.index('\n// SPLIT blocks',a)
    meta=cu[a:b]
    meta=once(meta,'__global__ void __launch_bounds__(256) route_live','__device__ __forceinline__ void p13_metadata')
    meta=once(meta,'ids[i]','__ldcg(ids+i)')
    meta=once(common,'// @@METADATA_DEVICE@@',meta)
    p3b=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='p3b_plan')
    plan=ast.get_source_segment(s,p3b)
    plan=once(plan,'def p3b_plan(data,base,native,streams):','def _p13_plan(data,base,native,streams,metadata):')
    plan=once(plan,'c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c)',
              'c,off,order,task0,slots,pos,task1,task2,_arrive=metadata')
    plan=once(plan,'tasks=[torch.empty_like(off) for _ in range(3)];slots=torch.empty((32,m),device=dev,dtype=torch.int32)',
              'tasks=[task0,task1,task2]')
    plan=once(plan,"ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids)",
              "ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous()")
    cpp='#include <cstdint>\n#include <vector>\n#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n'
    pool_cpp=cpp+'int p13_pool_launch(int,int64_t,int64_t,int64_t,int64_t,int,int,int,int,int,int,int,int64_t,int64_t,int64_t,int,std::vector<int64_t>);'
    meta_cpp=cpp+'void p13_meta_launch(std::vector<int64_t>,int64_t,int64_t);'
    return dict(P13_POOL_CUDA=pool,P13_POOL_CPP=pool_cpp,P13_META_CUDA=meta,P13_META_CPP=meta_cpp),plan


def main():
    result=json.loads((H/'result.json').read_text());assert result['build_fusion']
    src,plan=sources()
    source=base()+'\n'+'\n'.join(k+'='+repr(v) for k,v in src.items())+'\n'+plan+'\n'+(H/'fusion.py.inc').read_text()
    compile(source,'<P13 fusion>','exec')
    (H/'out/candidate.py').write_text(source)
    # Use the same reviewed lossless shipment format as S19. Keep the
    # expanded source separately for exact source/changed-unit audits.
    sys.path.insert(0,str(H.parent/'s16'))
    from pack import pack
    packed=pack(source)
    (H/'out/candidate_packed.py').write_text(packed)
    for k,v in src.items():(H/'out'/(k+'.cu' if k.endswith('CUDA') else k+'.cpp')).write_text(v)
    manifest=dict(base_sha256=sha(base()),sha256=sha(source),source='problems/muse_decode_layer/dev/p13_boundary/out/candidate.py',
                  shipped_sha256=sha(packed),shipped_source='problems/muse_decode_layer/dev/p13_boundary/out/candidate_packed.py',
                  oracle_job=result['job'],oracle_best=result['best_arm'],oracle_six_R=result['six_load'][result['best_arm']],
                  kernel_sha256={k:sha(v) for k,v in src.items()},small_rows=[1,8,16],large_rows=[128,256],
                  scope='Own route fusion: identical S19 raw pool producer with last-CTA norm/top8 epilogue at T1/8/16; row route plus last-CTA exact expert metadata at T128/256. T32 and b1/b2 unchanged.')
    (H/'fusion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
