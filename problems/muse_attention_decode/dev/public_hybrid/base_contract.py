"""Read-only S21 source identity and control-policy contract, no prepare/data."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];p=R/'problems/muse_decode_layer/cand_r3_best7_s21_final.py';raw=p.read_text();tree=ast.parse(raw)
# The shipped source is an owned compressed Python literal, never donor code.
assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='_source' for t in n.targets));blob=max((n.value for n in ast.walk(assign.value) if isinstance(n,ast.Constant) and isinstance(n.value,str)),key=len);s=lzma.decompress(base64.b85decode(blob)).decode();t=ast.parse(s)
def value(name):
 a=next(n for n in t.body if isinstance(n,ast.Assign) and any(getattr(x,'id',None)==name for x in n.targets));return ast.literal_eval(a.value)
selection=value('_PUBLIC_SELECTION');ps={};exec((H/'policy.py.inc').read_text(),ps);expected={tuple(map(int,c[:-1].split('x')))+(c[-1]=='s',):'operator' if a=='O' else 'lee' for c,a in ps['PICKS'].items()};assert all(expected[k]==v for k,v in selection.items()),(selection,expected);differences={str(k):dict(lab=expected[k],S21="C3 original") for k in expected if k not in selection}
ports=value('_PUBLIC_SOURCES');original=H.parent/'public_ports/out';assert all(ports[n]==(original/(n+'.py')).read_text() for n in ('operator','lee'));assert '_S19_BUILDS' in s
result=dict(base=str(p.relative_to(R)),sha256=hashlib.sha256(raw.encode()).hexdigest(),expanded_sha256=hashlib.sha256(s.encode()).hexdigest(),port_dispatch_differences=differences,port_sources_identical=True,prepare_build_hook=True,qualification='A/A2 use S19 at unported cells; S21 adds prior C3/native merge. Current S21 keeps C3 at d16F instead of the earlier OP point, and prior Rw4 at v32F. Remaining public modules match the measured picks exactly. P4 confirms whole-layer effects against those actual choices.')
(H/'base_contract.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
