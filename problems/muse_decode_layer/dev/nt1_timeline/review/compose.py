"""Compose only measured per-shape winners; the composition remains unmeasured."""
import ast,difflib,hashlib,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[4];sys.path.insert(0,str(D));import build
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def renamed(cell,tag):
 return cell.replace('smallm_',tag+'_').replace('_SMALLM_MODULE','_'+tag.upper()+'_MODULE').replace('SMALLM_FLAGS_BUILT',tag.upper()+'_FLAGS_BUILT').replace('SMALLM_VAR_',tag.upper()+'_')

def main():
 r=json.loads(Path(sys.argv[1]).read_text());assert r['measured'] and not r['invalid']
 rows={(v['M'],v['arm'],v['control']):v for v in r['summary'] if not v['invalid']}
 out=D/'combo';out.mkdir(exist_ok=True);base,_,_=build.original();b=base.index('def _s4_prepare_banks(')
 specs={};cells=[];selection={};cache={}
 for m in (1,8,16,32):
  available={arm:row for (mm,arm,ctrl),row in rows.items() if mm==m and ctrl=='S15' and row['ratio']<1 and not (arm=='BLOCK' and m not in (16,32))}
  best=min(available,key=lambda arm:available[arm]['ratio']) if available else 'S15'
  parents=[arm for arm in ('PAIR','SHARD2','SHARD4','SHARD8') if arm in available]
  parent=min(parents,key=lambda arm:available[arm]['ratio']) if parents else None
  features=([parent] if parent else [])+[arm for arm in ('BLOCK','CANCEL','PACK5') if arm in available]
  key=tuple(features)
  if not key:selection[str(m)]=dict(features=[],best_independent=best,prefix='smallm');continue
  if key not in cache:
   first=features[0];initial=build.source(first,rename=False)
   for arm in features[1:]:initial=build.source(arm,initial=initial,rename=False)
   cell,cuda=initial;tag='rvc_combo'+str(len(cache));cell=renamed(cell,tag);ast.parse(cell)
   name='COMBO'+str(len(cache));cp=out/(name.lower()+'_cell.py');cp.write_text(cell)
   cu=R/'runs/screen/p3a_review'/f'{name.lower()}.cu';cu.write_text(cuda)
   spec=dict(cell=str(cp.relative_to(R)),cell_sha256=sha(cell),cuda_sha256=sha(cuda),prefix=tag,features=features)
   specs[name]=spec;cells.append(cell);cache[key]=tag
  selection[str(m)]=dict(features=features,best_independent=best,prefix=cache[key],individual_ratios={arm:available[arm]['ratio'] for arm in features})
 wrapper='''
_combo_original_bind=smallm_bind
_combo_original_experts=smallm_experts
def rvc_combo_module():
'''+''.join(f'    {s["prefix"]}_module()\n' for s in specs.values())+'''
def rvc_combo_bind(state,flags_by_m):
'''+''.join(f'    {s["prefix"]}_bind(state,flags_by_m)\n' for s in specs.values())+'''    return state
def rvc_combo_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
'''+''.join(f'    if m=={m}:return {s["prefix"]}_experts(data,state,flags,stages,grid)\n' for m,s in selection.items())+'''    return _combo_original_experts(data,state,flags,stages,grid)
def smallm_bind(state,flags_by_m):
    _combo_original_bind(state,flags_by_m)
    return rvc_combo_bind(state,flags_by_m)
def smallm_experts(data,state,flags,stages=2,grid=264):
    return rvc_combo_experts(data,state,flags,stages,grid)
'''
 # The lab selector must call the exact original cell on a no-change shape.
 wrapper=wrapper.replace('return smallm_experts(data,state,flags,stages,grid)','return _combo_original_experts(data,state,flags,stages,grid)')
 candidate=base[:b]+'\n'.join(cells)+wrapper+'\n'+base[b:];ast.parse(candidate)
 path=R/'runs/screen/p3a_combo.py';path.write_text(candidate)
 patch=''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+str(build.BASE.relative_to(R)),tofile='b/'+str(build.BASE.relative_to(R))))
 (out/'combo.patch').write_text(patch);(out/'selector.py.inc').write_text(wrapper)
 report=dict(receipt=r['job'],selection=selection,variants=specs,candidate=str(path.relative_to(R)),candidate_sha256=sha(candidate),patch=str((out/'combo.patch').relative_to(R)),patch_sha256=sha(patch),base_sha256=build.BASE_SHA,status='unmeasured composition of per-shape nominees; no additive claim')
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 (out/'components_manifest.json').write_text(json.dumps(specs,indent=2)+'\n')
if __name__=='__main__':main()
