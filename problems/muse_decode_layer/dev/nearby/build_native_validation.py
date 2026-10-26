"""Prepare an independent native TASK gate for all65 S15/S15n audit shapes."""
import ast
import base64
import hashlib
import difflib
import json
import lzma
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[1]
out=H/'native_validation';out.mkdir(exist_ok=True)
base=(P/'cand_r3_best7_s15.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest().startswith('60254883')
near=(H/'nearby_payload.py').read_text()
near=near[:near.index('def lab(deadline,weights,state):')]
unguarded=(P/'cand_r3_best7_s15n.py').read_text()
assert hashlib.sha256(unguarded.encode()).hexdigest().startswith('ebceda91')
old="if cfg['batch']>1 and cfg['q_len']==8 and not cfg['window'] else None"
assert unguarded.count(old)==1
guarded=unguarded.replace(old,"if 1<cfg['batch']<=64 and cfg['q_len']==8 and not cfg['window'] else None")
assert hashlib.sha256(guarded.encode()).hexdigest().startswith('9cc90d2e')
(out/'guarded_s15n.py').write_text(guarded)
before,after=base.splitlines(True),guarded.splitlines(True)
edits=[(a,b,''.join(after[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if op!='equal']
near+='\nS15N_SHA='+repr(hashlib.sha256(guarded.encode()).hexdigest())+'\nS15N_EDITS='+repr(edits)+'\n'
near+="CASES += [dict(label='capacity-v'+str(b)+'-f',batch=b,q_len=8,ctx=129,layer='full',page=1,dist='soft',spread='fixed') for b in (64,65)]\n"
# Reuse the exact public checker already screened in S15L. Only the private
# reference namespace uses it; no imports of reference.py or checker outputs in
# a candidate's timed return path.
source=(P/'dev/library_chain/out/library_payload.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_BUNDLE')
bundle=eval(compile(ast.Expression(node.value),'bundle','eval'),dict(json=json,lzma=lzma,base64=base64))
enc=lambda s:base64.b85encode(lzma.compress(s.encode())).decode()
text=near+'\nVALIDATION_BASE=lzma.decompress(base64.b85decode('+repr(enc(base))+')).decode()\n'
text+='VALIDATION_CHECKER=lzma.decompress(base64.b85decode('+repr(enc(bundle['checker']))+')).decode()\n'
text+=(H/'native_validation.py.inc').read_text()
(out/'payload.py').write_text(text)
meta=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),checker_sha256=hashlib.sha256(bundle['checker'].encode()).hexdigest(),
    payload_sha256=hashlib.sha256(text.encode()).hexdigest(),candidate_sha256=hashlib.sha256(guarded.encode()).hexdigest(),shapes=67,seeds=2,arms=2,task_checks=268,
    decision='Send if S15n nearby timing/control supports retaining a range path; native TASK pass required before P4 promotion.')
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
