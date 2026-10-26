"""Static and sm_90a compile checks of the exact S19 expansion and each arm, dictionary-held kernels included."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(H),str(R/'tools'),str(H.parent/'s16')]
import local_eval,static_cache,build_lab,compose
static_cache.install(local_eval)


def units(name,source):
    path=H/'generated'/('audit_'+name+'.py');path.write_text(source)
    checks=local_eval.undefined_names(path)+local_eval.call_signatures(path)+local_eval.hopper_compile(path)
    for table in ('r2_KERNELS','wr_KERNELS'):
        for node in [n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==table for t in n.targets)]:
            for kind,code in ast.literal_eval(node.value).items():
                unit=H/'generated'/('audit_'+name+'_'+table+'_'+kind+'.py')
                unit.write_text('CUDA='+repr(code)+'\nload_inline(cuda_sources=CUDA,extra_cuda_cflags=["-O3","-std=c++20","-lineinfo"])\n')
                checks+=[dict(c,table=table,kernel=kind) for c in local_eval.hopper_compile(unit)]
    return [dict(c,unit=name) for c in checks]


def main():
    m=json.loads((H/'manifest.json').read_text());source=(R/m['candidate']).read_text()
    shipped=(R/m['shipped_source']).read_text();call=next(n for n in ast.walk(ast.parse(shipped)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
    assert lzma.decompress(base64.b85decode(ast.literal_eval(call.args[0]))).decode()==source and hashlib.sha256(source.encode()).hexdigest()==m['sha256']
    checks=units('S19',source)
    for name,(kw,_) in build_lab.ARMS.items():
        if kw is None:continue
        text=compose.build(**kw)[0];assert hashlib.sha256(text.encode()).hexdigest()==m['arms'][name]['sha256']
        checks+=units(name,text)
    result=dict(sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],checks=checks,passed=all(c['ok'] for c in checks))
    (H/'native_audit.json').write_text(json.dumps(result,indent=2)+'\n');assert result['passed'],[c for c in checks if not c['ok']]
    print('PASS',len(checks),'Python/native checks over S19 and',len(build_lab.ARMS)-1,'arms')
if __name__=='__main__':main()
