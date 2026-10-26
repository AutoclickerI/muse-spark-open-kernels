"""Per-shape S15 MoE shell point pieces; no row-wise promotion veto."""
import difflib,hashlib,json,pathlib
H=pathlib.Path(__file__).resolve().parent;R=H.parents[3];BASE='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
def apply(base,arm,shape):
 assert hashlib.sha256(base.encode()).hexdigest()==BASE and arm in ('R','F4','F8','C','E','P','PF') and shape in ('v16S','v16F','v32S','v32F')
 runtime=(H/'runtime_generated.py.inc').read_text();row=(int(shape[1:-1]),8,2048 if shape.endswith('S') else 0)
 suffix='''
_mt_point_base_run=_best7_run
_mt_point_base_expert=_best7_expert
def _best7_run(data,state,plan,prefetch=None):
    cfg=data['config'];plan=dict(plan,p7_point_shape=(cfg['batch'],cfg['q_len'],cfg['window']))
    return _mt_point_base_run(data,state,plan,prefetch)
def _best7_expert(pr,x,wts,lids,state,plan):
    if plan.get('p7_point_shape')!=MT_POINT_ROW:return _mt_point_base_expert(pr,x,wts,lids,state,plan)
    data=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,config=dict(local_start=0))
    for fn in mt_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'],MT_POINT_ARM)['steps']:out=fn()
    return out
'''
 return base+'\n'+runtime+'\nMT_POINT_ROW='+repr(row)+'\nMT_POINT_ARM='+repr(arm)+'\n'+suffix
if __name__=='__main__':
 base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();out=H/'out/point_handoffs';out.mkdir(parents=True,exist_ok=True);x=json.loads((H/'point_nominations.json').read_text())
 for row in x['nominations']:
  code=apply(base,row['arm'],row['shape']);tag=row['shape']+'_'+row['arm'];p=out/(tag+'.py');p.write_text(code)
  patch=''.join(difflib.unified_diff(base.splitlines(True),code.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15.py'));(out/(tag+'.patch')).write_text(patch)
  row.update(candidate_sha256=hashlib.sha256(code.encode()).hexdigest(),patch=str(out.relative_to(R)/(tag+'.patch')),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),applies_onto=BASE)
 (H/'point_nominations.json').write_text(json.dumps(x,indent=2)+'\n');print('P4 MoE shape pieces',len(x['nominations']))
