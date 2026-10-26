"""Concrete selected-arm patch onto S19; losing CUDA units are not embedded."""
import argparse,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'phases';ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,help='shape label -> measured point arm');ap.add_argument('--stem',default='candidate');a=ap.parse_args()
b=json.loads((o/'base_bundle.json').read_text());sel=json.loads(a.selection.read_text()) if a.selection else json.loads((o/'P4_HANDOFF.json').read_text())['preferred']
selection={tuple(map(int,k.removesuffix('f').split('x'))):v for k,v in sel.items()};units={}
for role in sorted(set(selection.values())):
 assert role in {'C3','T16','M64','U64'},'Only measured point nominations may enter a handoff'
 if role in ('M64','U64'):continue
 units[role]=((o/(role+'.cu')).read_text(),b['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])
body=(H/'phases_driver.py.inc').read_text()
for tag,keep in [('M',any(x.startswith(('M','U')) for x in selection.values())),('T',bool(set(selection.values())&{'C3','T8','T16'}))]:
 a0=body.index('    # BEGIN '+tag);z=body.index('    # END '+tag,a0)+len('    # END '+tag)
 if not keep:body=body[:a0]+body[z:]
if not units:
 a0=body.index('def _p5_phase_mod(');z=body.index('def _p5_phase_call(',a0);body=body[:a0]+body[z:]
 body=body.replace('    for role in _P5_PHASE_SOURCES:_p5_phase_mod(role)\n','')
if not any(x.startswith('M') for x in selection.values()):
 a0=body.index('    for role in _P5_PHASE_SELECTION.values():');z=body.index('_p5_phase_original_prepare=',a0);body=body[:a0]+body[z:]
if not units and not any(x.startswith('M') for x in selection.values()):body=body.replace('def _p5_phase_setup(device):\n_p5_phase_original_prepare=', 'def _p5_phase_setup(device):\n    pass\n_p5_phase_original_prepare=')
s=b['source']+'\n_P5_PHASE_SELECTION='+repr(selection)+'\n_P5_PHASE_SOURCES='+repr(units)+'\n'+body
compile(s,'S19_P5_phases','exec');p=o/(a.stem+'.py');p.write_text(s)
patch=''.join(difflib.unified_diff(b['source'].splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s19_final.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s19_final.py'))
(o/(a.stem+'.patch')).write_text(patch)
meta=dict(base_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),source_sha256=hashlib.sha256(s.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),selection=sel,units={k:hashlib.sha256(v[0].encode()).hexdigest() for k,v in units.items()},runtime='Same S19 partial-sum cast on q1; same qk_prep on M/U; metadata initialized from literal shape constants in prepare only. Only nominated full-attention rows are dispatched; no losing CUDA units are embedded.')
(o/(a.stem+'_manifest.json')).write_text(json.dumps(meta,indent=2)+'\n');print(meta)
