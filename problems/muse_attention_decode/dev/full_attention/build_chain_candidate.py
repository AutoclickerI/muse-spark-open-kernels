"""Only selected native points and shape schedules; independent patch onto S19."""
import argparse,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'chain'
ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--stem',default='candidate');a=ap.parse_args()
b=json.loads((H/'phases/base_bundle.json').read_text());sel=json.loads(a.selection.read_text());fa={};vq={};picks={}
points=json.loads((o/'result.json').read_text());assert points['measured']
admitted={(p['shape'],p['arm']) for p in points['points'] if p['nominated']}
assert all((k,v) in admitted for k,v in sel.items()),'Only correct measured point nominations enter a handoff'
for k,v in sel.items():
 t=tuple(map(int,k[:-1].split('x')))+(k[-1]=='s',)
 if t in ((16,8,False),(32,8,False)):
  assert v in ('R','Rw4','F4','F8','G4','G8');vq[t]=v
 else:
  assert v[0] in ('M','R');picks[t]=int(v[1:].split('w')[0])
  if v[0]=='R':fa[t]=v
s=b['source']
if picks:s+='\n'+(H/'chain_sched.py.inc').read_text();s=s.replace('_P5B_FA_PICKS={}', '_P5B_FA_PICKS='+repr(picks),1)
# Restrict the shape policy to the measured selected calls; other FA3 calls retain
# their original owned implementation. P5 native paths are never intercepted.
s=s.replace("if (c['batch'],c['q_len'],bool(c['window']),s) in _P5B_FA_META:","if (c['batch'],c['q_len'],bool(c['window'])) in _P5B_FA_PICKS and (c['batch'],c['q_len'],bool(c['window']),s) in _P5B_FA_META:")
if fa or any(v in ('R','Rw4') for v in vq.values()):
 runtime=(H/'chain_fa_runtime.py.inc').read_text()
 if not fa:
  # Only a native VQ8 row merge needs the reducer; do not ship an unused FA3 ABI.
  runtime=runtime[:runtime.index('def _p5b_merge_main')]+runtime[runtime.index('_p5b_merge_original_prepare='):]
  runtime=runtime.replace("    import sgl_kernel.flash_ops\n",'')
  start=runtime.index('    _P5B_BRIDGE=load_inline');end=runtime.index('    _P5B_REDUCE=load_inline',start);runtime=runtime[:start]+runtime[end:]
 s+='\n_P5B_MERGE_SELECTION='+repr(fa)+'\n_P5B_BRIDGE_SRC='+repr((o/'fa.cpp').read_text() if fa else '')+'\n_P5B_REDUCE_SRC='+repr((o/'reduce.cu').read_text())+'\n'+runtime
if vq:
 units={('N' if v in ('R','Rw4') else v[0]):(o/(('N' if v in ('R','Rw4') else v[0])+'.cu')).read_text() for v in vq.values()}
 if not any(v in ('R','Rw4') for v in vq.values()):
  body=(H/'chain_vq_runtime.py.inc').read_text();a0=body.index("    if role=='N':\n");z=body.index('    return out',a0);body=body[:a0]+body[z:]
 else:body=(H/'chain_vq_runtime.py.inc').read_text()
 s+='\n_P5B_VQ_SELECTION='+repr(vq)+'\n_P5B_VQ_SOURCES='+repr(units)+'\n'+body
compile(s,'S19_chain','exec');(o/(a.stem+'.py')).write_text(s)
patch=''.join(difflib.unified_diff(b['source'].splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s19_final.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s19_final.py'));(o/(a.stem+'.patch')).write_text(patch)
meta=dict(base_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),source_sha256=hashlib.sha256(s.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),selection=sel,scope='Only selected FA3 launch structure or full VQ8 merge. No P5 native/query/RAW2 changes.')
(o/(a.stem+'_manifest.json')).write_text(json.dumps(meta,indent=2)+'\n');print(meta)
