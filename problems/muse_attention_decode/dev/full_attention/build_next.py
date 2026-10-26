"""One whole-layer follow-up for the two independent reset outcomes."""
import ast
import base64
import difflib
import hashlib
import json
import lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=R/'problems/muse_decode_layer';out=H/'next';out.mkdir(exist_ok=True)
base=(P/'cand_r3_best7_s15.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('60254883')
sources={'V':(H/'pvsum/candidate.py').read_text(),'V2':(H/'pvsum/candidate.py').read_text(),
         'G':(H/'decode/out/s15_gather_16_32.py').read_text()}
specs={}
for name,s in sources.items():
    a,b=base.splitlines(True),s.splitlines(True)
    edits=[(lo,hi,''.join(b[c:d])) for op,lo,hi,c,d in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if op!='equal']
    specs[name]=dict(edits=edits,sha256=hashlib.sha256(s.encode()).hexdigest())
raw=(P/'dev/library_chain/out/library_payload.py').read_text()
n=next(n for n in ast.parse(raw).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_BUNDLE')
b=eval(compile(ast.Expression(n.value),'bundle','eval'),dict(json=json,lzma=lzma,base64=base64))
bundle=dict(arms=specs,inputs=b['inputs'],checker=b['checker'],probe=(H/'pvsum_probe.cu.inc').read_text())
enc=base64.b85encode(lzma.compress(json.dumps(bundle,separators=(',',':')).encode())).decode()
body=(P/'dev/library_chain/lab.py.inc').read_text()
body=body.replace("skip={'NF_PORT_SELECTED','_VQ8_ROWS'}","skip={'NF_PORT_SELECTED','_VQ8_ROWS'}\n    if ns['VQ8_CUDA']!=base['VQ8_CUDA']:skip.add('_VQ8_MOD')")
a=body.index('    # Exactly one additional NF O preparation');z=body.index('    a=torch.cuda.Event(enable_timing=True)',a)
body=body[:a]+(H/'next_native.py.inc').read_text()+body[z:]
a=body.index('    rows={};samples={};fail=[];agreements={};task_checks=0;native_proof=');z=body.index('    for batch,length,window in CASES:',a)
body=body[:a]+'''    rows={};samples={};fail=[];agreements={};task_checks=0
    native_proof={name:dict(cuda_sha256=hashlib.sha256(ns['VQ8_CUDA'].encode()).hexdigest(),gather='_dcd_attention' in ns)
                  for name,ns in spaces.items() if name not in ('A','A2')}
    native_proof['probes']=probe_rows
'''+body[z:]
body=body.replace('META r=arm/S15;L=all;no_*=L minus replacement;A2=A:A;8 seeds,8 cold graph replays',
    'META r=arm/S15;V=PV-group denominator,V2=V duplicate,G=vector page-ID gather,A2=A:A;8 seeds,8 cold graph replays;actual TASK gates')
body=body.replace('elapsed=round(time.monotonic()-started,2),complete=',
    'seed_us={f\'{b}x{l}{"s" if w else "f"}\':v for (b,l,w),v in samples.items()},'
    'elapsed=round(time.monotonic()-started,2),complete=')
text='import torch,time,statistics,math,json,gc,ast,hashlib,linecache,types,base64,lzma\n'
text+='_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(enc)+')))\n'+body
(out/'payload.py').write_text(text)
host=(P/'dev/s15/s15ab_host.py').read_text()
host+='\n_p5b_entry_kernel=globals()["kernel_fn"]\ndef kernel_fn(data,state):\n    return _p5b_entry_kernel(data,state)\n'
(out/'host.py').write_text(host)
meta=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),arms={k:v['sha256'] for k,v in specs.items()},
    payload_sha256=hashlib.sha256(text.encode()).hexdigest(),shapes=12,loads=6,seeds=8,replays=8,arms_including_controls=5,
    task_checks=480,probe='native E4M3 P*ones plus alpha rescale, repeated identical V/V2 executable',
    resources='V adds16B/thread main spills at168registers; no production C75 warnings; unchanged combine resources')
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
