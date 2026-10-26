"""Pack exact expanded candidates using S19's checked single-file wrapper."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];m=json.loads((D/'manifest.json').read_text());base=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text();blob=next(n.args[0].value for n in ast.walk(ast.parse(base)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');old=lzma.decompress(base64.b85decode(blob)).decode();assert hashlib.sha256(old.encode()).hexdigest()=='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a'
for arm,v in m.items():
 p=R/v['candidate'];s=p.read_text();assert hashlib.sha256(s.encode()).hexdigest()==v['candidate_sha256'];new=base.replace(blob,base64.b85encode(lzma.compress(s.encode(),preset=9)).decode(),1).replace(v['base_sha256'],v['candidate_sha256']);out=p.with_name(p.stem+'_packed.py');out.write_text(new);v.update(expanded_candidate=v['candidate'],expanded_candidate_sha256=v['candidate_sha256'],shipped=str(out.relative_to(R)),shipped_sha256=hashlib.sha256(new.encode()).hexdigest())
(D/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
