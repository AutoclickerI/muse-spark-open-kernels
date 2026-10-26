"""Exact licensed patch onto S21 +9b043487, selected only after native graph evidence."""
import argparse,difflib,hashlib,json
from pathlib import Path
from build import control,H,O,P
ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path);ap.add_argument('--stem',default='all_F');args=ap.parse_args()
b=json.loads((O/'bundle.json').read_text())
if args.selection:selection=json.loads(args.selection.read_text())
else:selection={'d1S':'V','d1F':'V','d16S':'F','d32S':'F','v1S':'F','v1F':'F','v16S':'F','v16F':'F','v32S':'F'}
dispatch={};sources={}
for cell,arm in selection.items():
 assert arm in ('D','V','F'),'S19 restores are reported separately for P4 composition'
 batch=int(cell[1:-1]);length=1 if cell[0]=='d' else 8;key=batch,length,cell[-1]=='S';dispatch[key]=arm
 if key in ((1,1,True),(1,1,False)):continue
 name=('op_' if key==(1,8,True) else 'lee_')+arm;sources[name]=b['modules'][name]
 if arm=='D' and key not in ((32,1,True),(1,8,True)):sources['pack']=b['modules']['pack']
base=control();source=base+'\n_PV_SELECTION='+repr(dispatch)+'\n_PV_SOURCES='+repr(sources)+'\n'
if selection.get('d1S'):
 source+='\n_s16_global_C128_cs_CUDA='+repr(b['native_cs'])+'\n_s16_global_C128_cs__MOD={}\n'
if selection.get('d1F'):source+='\nPARTIAL_CUDA='+repr(b['native_partial'])+'\n_PP_MOD={}\n'
source+=(H/'runtime.py.inc').read_text();compile(source,'candidate','exec')
# The control already includes complete public notices, verified byte-for-byte.
notices=(P/'NOTICES.txt').read_text();assert repr(notices) in source
path='problems/muse_decode_layer/cand_r3_best7_s21_final.py'
patch=''.join(difflib.unified_diff(base.splitlines(True),source.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
(O/(args.stem+'.py')).write_text(source);(H/(args.stem+'.patch')).write_text(patch)
meta=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),selection=selection,licenses_sha256=hashlib.sha256(notices.encode()).hexdigest(),status='ready, unmeasured until production-graph receipt; not a P4 winner')
(H/(args.stem+'_manifest.json')).write_text(json.dumps(meta,indent=2)+'\n');print(meta)
