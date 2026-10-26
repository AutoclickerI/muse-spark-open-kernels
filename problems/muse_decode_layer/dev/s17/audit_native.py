"""Audit the exact packed expansion and the additional dictionary-held P5 CUDA."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(H.parent/'s16')]
import local_eval,static_cache
static_cache.install(local_eval)

def main():
    late='--late' in sys.argv
    m=json.loads((H/'generated'/('lab_late_manifest.json' if late else 'lab_manifest.json')).read_text());p=R/m['candidate'];source=p.read_text()
    shipped=(R/m['shipped_source']).read_text();call=next(n for n in ast.walk(ast.parse(shipped)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
    expanded=lzma.decompress(base64.b85decode(ast.literal_eval(call.args[0]))).decode();assert source==expanded and hashlib.sha256(source.encode()).hexdigest()==m['sha256']
    checks=local_eval.undefined_names(p)+local_eval.call_signatures(p)+local_eval.hopper_compile(p)
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='r2_KERNELS' for t in n.targets)]
    assert len(nodes)==1
    kernels=ast.literal_eval(nodes[0].value);units={}
    for name,code in kernels.items():
        tag=hashlib.sha256(code.encode()).hexdigest();unit=H/'generated'/('audit_p5_'+name+'.py')
        unit.write_text('CUDA='+repr(code)+'\nload_inline(cuda_sources=CUDA,extra_cuda_cflags=["-O3","-std=c++20","-lineinfo"])\n')
        checks+=local_eval.hopper_compile(unit);units[name]=tag
    result=dict(sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],P5_units=units,checks=checks,passed=all(c['ok'] for c in checks))
    (H/('native_late_audit.json' if late else 'native_audit.json')).write_text(json.dumps(result,indent=2)+'\n');assert result['passed'],checks
    print('PASS',len(checks),'Python/exact native units',units)
if __name__=='__main__':main()
