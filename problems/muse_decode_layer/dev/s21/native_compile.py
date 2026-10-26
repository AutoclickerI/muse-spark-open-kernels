"""Compile dictionary-held chain CUDA that local_eval's literal-unit scan cannot see."""
import ast,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parents[3]/'tools'))
from local_eval import hopper_compile

def main():
    tree=ast.parse((H/'generated/s21.py').read_text())
    sources=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_P5B_VQ_SOURCES')
    code='\n'.join('CHAIN_'+k+'='+repr(v) for k,v in sources.items())+'\n'
    code+='\n'.join("load_inline(cuda_sources=CHAIN_"+k+",extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr'])" for k in sources)+'\n'
    ports=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_PUBLIC_SOURCES')
    cuda=next(ast.literal_eval(n.value) for n in ast.parse(ports['lee']).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA')
    code+='LEE_CUDA='+repr(cuda)+"\nload_inline(cuda_sources=LEE_CUDA,extra_cuda_cflags=['-O3','-lineinfo','-DRSW=4','-DPRS=4','-DDRS=4','-DRBF=5'])\n"
    path=H/'generated/chain_native_units.py';path.write_text(code)
    checks=hopper_compile(path)
    assert len(checks)==len(sources)+1,(checks,sources.keys())
    (H/'generated/chain_native_compile.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2));assert all(x['ok'] for x in checks)
if __name__=='__main__':main()
