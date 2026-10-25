"""Two-step dead-init ablation on exact S6 mixed_parallel; original modes stay intact."""
import ast
import hashlib
import json
from pathlib import Path
import sys

H=Path(__file__).resolve().parent;P=H.parent;R=H.parents[4]
sys.path[:0]=[str(P),str(R/'tools')]
from build_prefill_lab import subset
from okc_run import rejected_words


def value(text,key):
    return next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==key for t in n.targets))


def main():
    path=R/'problems/muse_decode_layer/cand_r3_best7_s6.py';text=path.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()=='4c98f12fc7fab9730b90d3bae2f7f014ce85bae35439fb688282d98bd167c59f'
    original=value(text,'FX_CUDA');a=original.index('__global__ void route_init(');b=original.index('template<int NT,int KT,int S,int SPLIT,bool FIRST',a)
    route=original[a:b].replace('void route_init(','void route_live(')
    dead=''' for(int i=all;i<clear_count;i+=step)state[i]=0;
 for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);
'''
    assert dead in route;route=route.replace(dead,'')
    a=route.index(' for(int token=t;token<M;token+=256)');b=route.index('\n __syncthreads();',a);route=route[:a]+route[b:]
    assert 'state[' not in route and 'reinterpret_cast<uint4*>(out)' not in route
    extra='\nnamespace mrinit {\n'+route+'\n}\n'+'''
void ri_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear,int64_t stage,int64_t stream){
 WX_CHECK(stage==1||stage==2);mrinit::route_live<<<stage==1?128:1,256,0,(cudaStream_t)stream>>>(
 (int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear);WX_CUDA_CHECK(cudaGetLastError());}
'''
    (H/'route.cu').write_text('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <vector>\n'+extra[:extra.index('void ri_route')])
    engine=subset(text,('wi_prepare','wi_plan','wi_call'))
    tree=ast.parse(engine);tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom,ast.Try))]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='FX_CUDA' for t in n.targets):n.value=ast.Constant(original+extra)
        if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='WI_CPP' for t in n.targets):n.value=ast.Constant(ast.literal_eval(n.value)+'void ri_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);')
        if isinstance(n,ast.FunctionDef) and n.name=='fx_module':
            for node in ast.walk(n):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='load_inline':
                    kw={v.arg:v for v in node.keywords};kw['functions'].value.elts.append(ast.Constant('ri_route'))
                    kw['name'].value=ast.parse("'route_s6_'+hashlib.sha256(FX_CUDA.encode()).hexdigest()[:12]",mode='eval').body
            n.body.insert(0,ast.parse("ri_need(40.,'native compile')").body[0])
    imports='import functools\nimport hashlib\nimport torch\nimport triton\nimport triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\n'
    adapter=imports+ast.unparse(ast.fix_missing_locations(tree))+'\n'+(H/'adapter.py.inc').read_text()
    (H/'adapter.py').write_text(adapter)
    helpers=subset((R/'prof/lab/moe_w2fix_m256.py').read_text(),('dc_weights','dc_fixtures','dc_oracles','MWGraph'))
    body=helpers+'\n'+adapter+'\n'+(H/'driver.py.inc').read_text();assert not rejected_words(body),rejected_words(body)
    out=R/'prof/lab/moe_route_init_m256.py';out.write_text(body)
    meta=dict(payload=str(out.relative_to(R)),sha256=hashlib.sha256(body.encode()).hexdigest(),S6_sha256=hashlib.sha256(text.encode()).hexdigest(),original_FX_CUDA_sha256=hashlib.sha256(original.encode()).hexdigest(),augmented_FX_CUDA_sha256=hashlib.sha256((original+extra).encode()).hexdigest(),live_route_sha256=hashlib.sha256(route.encode()).hexdigest(),stages=['S6 original grid128','dead stores removed grid128','same live route kernel grid1'],not_before='2026-10-25T09:01:00+09:00')
    (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['payload'],meta['sha256'],len(body))


if __name__=='__main__':main()
