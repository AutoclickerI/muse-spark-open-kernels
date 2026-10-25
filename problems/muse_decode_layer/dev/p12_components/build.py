"""Package exact S15 paths with board ABI adapters; no layer source changes."""
import base64,hashlib,json,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
p=R/'problems/muse_decode_layer/cand_r3_best7_s15.py';src=p.read_text();sha=hashlib.sha256(src.encode()).hexdigest()
assert sha.startswith('60254883'),sha
notice=(R/'problems/muse_bf16_proj/dev/public_nf/LICENSE.NathenFernandes').read_text()
pre='\n'.join('# '+l for l in ('Nathen Fernandes; github.com/NathenFernandes/muse-bf16-proj @ 86aff75; MIT\n'+notice).splitlines())
body=src
rows=[]
for role,problem in (('moe','muse_moe_w4a16'),('attention','muse_attention_decode'),('projection','muse_bf16_proj')):
 out=H/('cand_s15_'+role+'.py');s='# okc: problem='+problem+'\n'+pre+'\n'+body+'\n'+(H/(role+'.py.inc')).read_text()
 compile(s,str(out),'exec');assert len(s.encode())<1_000_000
 out.write_text(s);rows.append(dict(role=role,problem=problem,file=str(out.relative_to(R)),sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode())))
(H/'manifest.json').write_text(json.dumps(dict(base=str(p.relative_to(R)),base_sha256=sha,candidates=rows),indent=2)+'\n')
for row in rows:print(row['role'],row['bytes'],row['sha256'][:8])
