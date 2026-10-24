from pathlib import Path
import ast,sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval
p=ROOT/'problems/muse_decode_layer/cand_r3_omni.py';src=p.read_text();t=ast.parse(src)
n=next(n for n in t.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='OMNI_MIXED_CUDA')
tmp=Path('/tmp/omni_mixed_compile_input.py');tmp.write_text('OMNI_MIXED_CUDA = '+repr(ast.literal_eval(n.value))+'''\ndef loader():
    load_inline(cuda_sources=OMNI_MIXED_CUDA, extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/deep_gemm/include'], extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr'])
''')
import subprocess
run=subprocess.run
def debug_run(*a,**kw):
    result=run(*a,**kw)
    if result.returncode:Path('/tmp/omni_mixed_nvcc_errors.log').write_text(result.stderr)
    return result
subprocess.run=debug_run
rows=local_eval.hopper_compile(tmp)
print(rows,flush=True)
assert all(r['ok'] for r in rows)
