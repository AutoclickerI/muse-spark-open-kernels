"""Audit the expanded exact shipped S16, including native code hidden by packing."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval,static_cache
static_cache.install(local_eval)


def main():
    manifest=json.loads((H/'generated/lab_manifest.json').read_text())
    file=R/'problems/muse_decode_layer'/('cand_r3_best7_s16_'+manifest['sha256'][:12]+'.py')
    tree=ast.parse(file.read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
    expanded=lzma.decompress(base64.b85decode(ast.literal_eval(call.args[0])))
    assert hashlib.sha256(expanded).hexdigest()==manifest['sha256']
    target=R/manifest['candidate'];assert target.read_bytes()==expanded
    rows=local_eval.undefined_names(target)+local_eval.call_signatures(target)+local_eval.hopper_compile(target)
    report=dict(expanded_sha256=manifest['sha256'],shipped_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),checks=rows,passed=all(r['ok'] for r in rows))
    (H/'generated/expanded_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert len(rows)>2 and report['passed'],rows
    print('PASS exact packed expansion and',len(rows),'expanded Python/native checks')
if __name__=='__main__':main()
