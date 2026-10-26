"""Hopper-compile every CUDA source of the exact S18 expansion and each leave-out arm."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(H.parent/'s16')]
import local_eval,static_cache
static_cache.install(local_eval)


def units(name,source):
    """The whole-file check misses dictionary-held P5 kernels; compile them as their own units."""
    path=H/'generated'/('audit_'+name+'.py');path.write_text(source)
    checks=local_eval.undefined_names(path)+local_eval.call_signatures(path)+local_eval.hopper_compile(path)
    (node,)=[n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='r2_KERNELS' for t in n.targets)]
    for kind,code in ast.literal_eval(node.value).items():
        unit=H/'generated'/('audit_'+name+'_p5_'+kind+'.py')
        unit.write_text('CUDA='+repr(code)+'\nload_inline(cuda_sources=CUDA,extra_cuda_cflags=["-O3","-std=c++20","-lineinfo"])\n')
        checks+=local_eval.hopper_compile(unit)
    return [dict(c,unit=name) for c in checks]


def main():
    m=json.loads((H/'manifest.json').read_text());source=(R/m['candidate']).read_text()
    shipped=(R/m['shipped_source']).read_text();call=next(n for n in ast.walk(ast.parse(shipped)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
    assert lzma.decompress(base64.b85decode(ast.literal_eval(call.args[0]))).decode()==source and hashlib.sha256(source.encode()).hexdigest()==m['sha256']
    checks=units('S18',source);lines=source.splitlines(True)
    for name,arm in m['arms'].items():
        if name=='INC':continue
        restored=list(lines)
        for a,b,text in reversed(arm['edits']):restored[a:b]=[text]
        text=''.join(restored);assert hashlib.sha256(text.encode()).hexdigest()==arm['sha256']
        checks+=units(name,text)
    result=dict(sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],checks=checks,passed=all(c['ok'] for c in checks))
    (H/'native_audit.json').write_text(json.dumps(result,indent=2)+'\n');assert result['passed'],[c for c in checks if not c['ok']]
    print('PASS',len(checks),'Python/native checks over S18 and',len(m['arms'])-1,'leave-out arms')
if __name__=='__main__':main()
