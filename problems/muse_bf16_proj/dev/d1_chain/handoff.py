"""Rebase each historical d1 point nomination onto exact S15 for P4 confirmation."""
import difflib,hashlib,importlib.util,json,pathlib
H=pathlib.Path(__file__).resolve().parent;R=H.parents[3];S15='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
spec=importlib.util.spec_from_file_location('p7_d1_point_builder',H/'patch.py');recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
def apply(base,arm,shape):
 assert hashlib.sha256(base.encode()).hexdigest()==S15
 assert shape in ('d1S','d1F') and arm in ('D','W','L','S','W1','WR','W2')
 roles={'W1':('b1',),'WR':('route',),'W2':('b2',)}.get(arm,('b1','route','b2'))
 mode={'D':'weights','W':'weights','L':'late_prew','S':'stream','W1':'weights','WR':'weights','W2':'weights'}[arm]
 assert arm!='D' or shape=='d1F','D does not change d1 sliding'
 rows=tuple((1,1,2048 if shape=='d1S' else 0,role) for role in roles)
 cuda=recipe.source(base,mode);runtime=recipe.previous.RUNTIME.replace('__GP_CUDA__',repr(cuda),1).replace('__GP_ROWS__',repr(rows),1)
 a='    _gate_split_prepare(state,device)\n';assert base.count(a)==1;base=base.replace(a,a+'    gp_module()\n',1)
 a='    _best7_prepare_settings(state,weights,device)\n';assert base.count(a)==1;base=base.replace(a,a+"    for key,plan in state['best7_plans'].items():plan['gp_key']=key\n",1)
 return base+'\n\n'+runtime
if __name__=='__main__':
 base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();out=H/'out/point_handoffs';out.mkdir(parents=True,exist_ok=True);x=json.loads((H/'point_nominations.json').read_text())
 for row in x['nominations']:
  if row['unchanged_duplicate']:continue
  code=apply(base,row['arm'],row['shape']);tag=row['shape']+'_'+row['arm'];p=out/(tag+'.py');p.write_text(code)
  patch=''.join(difflib.unified_diff(base.splitlines(True),code.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'))
  (out/(tag+'.patch')).write_text(patch);row.update(candidate_sha256=hashlib.sha256(code.encode()).hexdigest(),patch=str(out.relative_to(R)/(tag+'.patch')),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),applies_onto=S15)
 (H/'point_nominations.json').write_text(json.dumps(x,indent=2)+'\n');print('P4 handoffs',sum(not r['unchanged_duplicate'] for r in x['nominations']))
