"""Freeze repaired K64 configurations and a K128 control within one lab share."""
import ast
import hashlib
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_prefill_lab import subset
from source import source,CONFIGS
from native_abi import lean,PRELUDE,CPP_PRELUDE
ROOT=HERE.parents[4]


def rename(code,prefix,ids):
    a=code.index('#define FT_CASES(F)');b=code.index('\n}',a)
    cases='#define FT_CASES(F) \\\n '+' \\\n '.join(f'F({i},{nt},{kt},{st},{sk},{str(ad).lower()})' for i,(nt,kt,st,sk,ad) in enumerate(CONFIGS) if i in ids)
    code=code[:a]+cases+code[b:]
    return code.replace('namespace mw128','namespace '+prefix).replace('mw128::',prefix+'::').replace('m128_',prefix+'_')


def pointer_bindings(s):
    s=lean(s);a=s.index('void rot64_route(');b=s.index('void rot64_run(',a)
    s=s[:a]+'''void rot64_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){
 rot64::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());
}
'''+s[b:]
    s=s.replace('std::vector<torch::Tensor> v','std::vector<int64_t> v')
    s=s.replace('int64_t grid,int64_t start){','int64_t grid,int64_t start,int64_t M,int64_t cap,int64_t stream_value){')
    s=s.replace('int M=v[11].size(0),cap=v[9].size(0);auto stream=at::cuda::getCurrentCUDAStream();','auto stream=(cudaStream_t)stream_value;')
    s=re.sub(r'v\[(\d+)\]\.data_ptr<([^>]+)>\(\)',lambda m:'('+m[2]+'*)v['+m[1]+']',s)
    s=re.sub(r'v\[(\d+)\]\.data_ptr\(\)',lambda m:'(void*)v['+m[1]+']',s)
    assert 'torch::' not in s and 'at::' not in s
    return PRELUDE+s


def main():
    prior=ROOT/'prof/lab/moe_m128_one_read.py';tree=ast.parse(prior.read_text())
    old=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='M128_CUDA' for t in n.targets))
    assert hashlib.sha256(old.encode()).hexdigest()=='14893c8304d7c9a9d0b0e382ced8f949c4429cf9c7e4e9af3091deec9c8be47f'
    native=pointer_bindings(rename(source(),'rot64',(3,4,5)))
    helpers=subset(prior.read_text(),('dc_weights','dc_fixtures','dc_oracles','mw_repack_planes','MWGraph'))
    # subset keeps imports, but these kernels need no Humming runtime.
    t=ast.parse(helpers);t.body=[n for n in t.body if not isinstance(n,(ast.Import,ast.ImportFrom)) or
        all(not a.name.startswith(('humming','sglang')) for a in n.names) and not (isinstance(n,ast.ImportFrom) and (n.module or '').startswith(('humming','sglang')))]
    helpers=ast.unparse(t)
    # The common packed bank and Python launch ABI are unchanged; no loaded module is edited.
    ft=ast.parse((HERE.parent/'fused.py').read_text())
    defs='\n'.join(ast.unparse(n) for n in ft.body if isinstance(n,ast.FunctionDef) and n.name in ('ft_prepare','ft_from_packed'))
    defs=defs.replace('mod=ft_module()', 'mod=RepairABI("rot64")')
    cpp=CPP_PRELUDE+'int64_t rot64_init();std::vector<int64_t> rot64_info(int64_t,int64_t);void rot64_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void rot64_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'
    funs=[p+'_'+f for p in ('rot64',) for f in ('init','info','route','run')]
    adapter=ast.parse((HERE/'adapter.py').read_text());work=next(n for n in adapter.body if isinstance(n,ast.FunctionDef) and n.name=='work')
    work.name='repair_plan';work.body[-1]=ast.parse('return {"steps":(route,first,second),"v13":v13,"v2":v2}').body[0]
    defs+='\n'+ast.unparse(work)
    out=ROOT/'prof/lab/moe_m128_k64_repair.py'
    out.write_text(helpers+'\nimport functools\nimport hashlib\nfrom torch.utils.cpp_extension import load_inline\nCONFIGS='+repr(CONFIGS)+'\nREPAIR_CUDA='+repr(native)+'\nREPAIR_CPP='+repr(cpp)+'\nREPAIR_FUNCTIONS='+repr(funs)+'\n'+defs+'\n'+(HERE/'repair_driver.py.inc').read_text())
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)


if __name__=='__main__':main()
