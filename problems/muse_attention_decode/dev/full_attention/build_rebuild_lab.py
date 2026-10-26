"""One S15 paired whole-layer payload for the three independent review targets."""
import argparse,base64,difflib,hashlib,json,lzma
from pathlib import Path
from lab_sources import public_bundle,host as make_host
H=Path(__file__).resolve().parent;R=H.parents[3];P=R/'problems/muse_decode_layer'
ap=argparse.ArgumentParser();ap.add_argument('--reuse',action='store_true');args=ap.parse_args()
out=H/('reuse' if args.reuse else 'rebuild');out.mkdir(exist_ok=True)
names=['kmajor','kmajor_reuse'] if args.reuse else ['ready','ready_elide','reg','parity','kmajor']
checks=(len(names)+2)*(12*8+13*2)
base=(P/'cand_r3_best7_s15.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('60254883')
meta={};arms={}
for name in names:
 s=(H/'rebuild'/(name+'.py')).read_text();a,b=base.splitlines(True),s.splitlines(True)
 arms[name]=dict(edits=[(lo,hi,''.join(b[c:d])) for op,lo,hi,c,d in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if op!='equal'],sha256=hashlib.sha256(s.encode()).hexdigest())
bundle=public_bundle(P);bundle['arms']=arms;bundle['probe']=(H/'kmajor_probe.cu.inc').read_text()
enc=base64.b85encode(lzma.compress(json.dumps(bundle,separators=(',',':')).encode())).decode()
body=(P/'dev/library_chain/lab.py.inc').read_text().replace('6107923+7919*i',('128107923' if args.reuse else '116107923')+'+7919*i')
body=body.replace("skip={'NF_PORT_SELECTED','_VQ8_ROWS'}","skip={'NF_PORT_SELECTED','_VQ8_ROWS'}\n    if ns['VQ8_CUDA']!=base['VQ8_CUDA']:skip.add('_VQ8_MOD')")
body=body.replace("checker=ns_source('independent_TASK_checker',_BUNDLE['checker']).__dict__", "checker=ns_source('independent_TASK_checker',_BUNDLE['checker']).__dict__\n    if weights is None:\n        weights=checker['generate_weights'](checker['WEIGHTS_SEED'])\n        state=base['_s15_prepare'](weights)")
a=body.index('    # Exactly one additional NF O preparation');z=body.index('    a=torch.cuda.Event(enable_timing=True)',a);body=body[:a]+(H/'rebuild_native_setup.py.inc').read_text()+(H/'rebuild_preflight.py.inc').read_text()+body[z:]
body=body.replace('assert preflight_checks==182','assert preflight_checks=='+str(26*(len(names)+2)))
if args.reuse:
 body=body.replace("if name=='kmajor':", "if name in ('kmajor','kmajor_reuse'):")
 body=body.replace("cu=ns['ATTN_PORT_CUDA_SRC']+_BUNDLE['probe'];", "cu=ns['ATTN_PORT_CUDA_SRC']+_BUNDLE['probe']+'\\nint64_t kmajor_occupancy(){return occupancy_t<1,4,true>();}\\n';")
 body=body.replace('void kmajor_probe(int64_t,int64_t,int64_t,int64_t);', 'void kmajor_probe(int64_t,int64_t,int64_t,int64_t); int64_t kmajor_occupancy();')
 body=body.replace("'occupancy','kmajor_probe'", "'occupancy','kmajor_probe','kmajor_occupancy'")
a=body.index('native_proof={name:dict(');z=body.index('    for batch,length,window in CASES:',a)
body=body[:a]+'''native_proof={name:dict(vq8_sha256=hashlib.sha256(ns['VQ8_CUDA'].encode()).hexdigest(),q1_sha256=hashlib.sha256(ns['ATTN_PORT_CUDA_SRC'].encode()).hexdigest()) for name,ns in spaces.items()}
    native_proof['kmajor_probe']=probe_rows
    native_proof['official_TASK_checks']=preflight_checks
    task_checks=preflight_checks
'''+body[z:]
if args.reuse:
 body=body.replace("    native_proof['kmajor_probe']=probe_rows", "    native_proof['occupancy']={'A':base['_attn_port_ext']().occupancy(1),**{name:compiled[name].kmajor_occupancy() for name in ('kmajor','kmajor_reuse')}}\n    native_proof['kmajor_probe']=probe_rows")
body=body.replace('elapsed=round(time.monotonic()-started,2),complete=',
    'seed_us={f\'{b}x{l}{"s" if w else "f"}\':[{name:round(us,3) for name,us in row.items()} for row in v] for (b,l,w),v in samples.items()},recorded_us_precision=.001,elapsed=round(time.monotonic()-started,2),complete=')
body=body.replace('META r=arm/S15;L=all;no_*=L minus replacement;A2=A:A;8 seeds,8 cold graph replays',
    'META r=arm/S15; independent K/V-ready,elided redundant V wait,register split,even-odd softmax+split,K-major q1; A:A;8 fresh seeds8cold rounds;TASK only;all per-seed times')
text='import torch,time,statistics,math,json,gc,ast,hashlib,linecache,types,base64,lzma\n_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(enc)+')))\n'+body
(out/('reuse_attention.py' if args.reuse else 'structural_attention.py')).write_text(text)
host=make_host(base)+'\n_entry=globals()["kernel_fn"]\ndef kernel_fn(data,state):\n    return _entry(data,state)\n';(out/'host.py').write_text(host)
meta=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),arms={n:a['sha256'] for n,a in arms.items()},payload_sha256=hashlib.sha256(text.encode()).hexdigest(),shapes=12,seeds=8,task_checks=checks,replays=8,mode='decode import lab;private weights/state discarded;host calls remain exactS15')
(out/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
