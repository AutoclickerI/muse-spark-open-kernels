"""Build the P13 excluded-work oracle on the immutable S19 shipment."""
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[3]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def functions(path, names):
    s = path.read_text()
    return '\n'.join(ast.get_source_segment(s, n) for n in ast.parse(s).body
                     if isinstance(n, ast.FunctionDef) and n.name in names)


def base():
    frozen = json.loads((H.parent/'s19/frozen.json').read_text())
    packed = (R/frozen['file']).read_text()
    assert sha(packed) == frozen['shipped_sha256']
    node = next(n for n in ast.parse(packed).body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', '') == '_source' for t in n.targets))
    blob = next(n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant)
                and isinstance(n.value, str) and len(n.value) > 10000)
    source = lzma.decompress(base64.b85decode(blob)).decode()
    assert sha(source) == frozen['expanded_sha256']
    return source


def main():
    (H/'out').mkdir(exist_ok=True)
    spec = 'import math,functools\nimport torch\n'
    for name in ('nvfp4.py', 'checks.py', 'reference.py'):
        s = (R/'problems/muse_decode_layer'/name).read_text()
        spec += '\n'.join(ast.get_source_segment(s, n) for n in ast.parse(s).body
                          if isinstance(n, (ast.Assign, ast.AnnAssign, ast.FunctionDef)))+'\n'
    checker = functions(R/'problems/muse_attention_decode/dev/early_history/lab_body.inc', {'eh_checker'})
    timing = functions(H.parent/'s21/timing.py.inc', {'s21_telemetry', 's21_orders'})
    body = 'import base64,gc,hashlib,json,linecache,lzma,math,re,statistics,time,types\nimport torch\n'
    body += 'BASE='+repr(base())+'\nPUBLIC_SPEC={"torch":torch}\nexec('+repr(spec)+',PUBLIC_SPEC)\n'
    body += checker+'\nS21_NVML=None\nS21_TELEMETRY_ERROR=None\n'+timing+'\n'
    body += (H/'codec.py').read_text()+'\n'+(H/'oracle.py.inc').read_text()
    compile(body, '<P13-body>', 'exec')
    (H/'out/body.py').write_text(body)
    blob = base64.b85encode(lzma.compress(body.encode(), preset=9)).decode()
    payload = 'LAB_RESOURCES={"compile_s":190,"run_s":500,"output_bytes":13744,"context_poison":"none"}\n'
    payload += 'import base64,lzma,linecache\n_BODY=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n'
    payload += '_FN="<P13-boundary-route-oracle>"\nlinecache.cache[_FN]=(len(_BODY),None,_BODY.splitlines(True),_FN)\nexec(compile(_BODY,_FN,"exec"),globals())\n_P13_RUN=globals()["lab"]\ndef lab(deadline,weights,state=None):return _P13_RUN(deadline,weights,state)\n'
    (H/'out/p13_oracle.py').write_text(payload)
    manifest = dict(base_sha256=sha(base()), payload_sha256=sha(payload), body_sha256=sha(body),
                    checker_sha256=sha(spec), weight_seed=130001003, router_seed=130001009,
                    seeds=[160001021+104729*i for i in range(8)],
                    arms=['BASE', 'AA', 'ROUTE', 'BOUND', 'BOTH'],
                    fixture=dict(distribution='soft on both S/F',shard='exact public ref_front biased top8 busiest expert shard per fresh fixture'),
                    measurement_protocol=dict(name='palindromic-cohort-v1', rounds=10, seed_count=8,
                                              aggregation='arithmetic-mean-all-rounds', selective_retiming=False),
                    scope='Excluded-work oracle only. Ready route/norm and external expert metadata, b1/b2, or both, precomputed per fresh fixture; never a candidate.')
    (H/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
