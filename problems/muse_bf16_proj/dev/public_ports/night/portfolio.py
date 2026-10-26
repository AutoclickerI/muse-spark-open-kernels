"""Select parent in-graph points by mean six-load effect; no new promotion veto."""
import importlib.util,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
def select_parent(result):
 points=[];kept=[];dropped=[]
 for name,pts in result['manifest']['points'].items():
  if name=='PORTS':continue
  arm=result['arms'][name];row=dict(arm=name,points=pts,measured=arm['measured'],R=arm.get('R'))
  if arm['measured'] and arm['R'][0]>1:points.extend(pts);kept.append(row)
  else:dropped.append(row)
 pool={};pool_verdict={}
 for t in (16,32):
  load='d'+str(t);choices=[(1.,None)]
  for name in result['manifest']['pool']:
   arm=result['arms'][name]
   if arm['measured']:choices.append((arm['loads'][load]['R'][0],name))
  for row in kept:
   if row['points']==[[t,'pool_router','P']] or row['points']==[[t,'pool_router','O']]:choices.append((result['arms'][row['arm']]['loads'][load]['R'][0],row['arm']))
  _,best=max(choices,key=lambda x:x[0]);pool_verdict[t]=dict(selected=best,choices=choices)
  if best in result['manifest']['pool']:
   pool[t]=best;points=[p for p in points if p[:2]!=[t,'pool_router']]
 return dict(parent=result['job'],parent_receipt_job=result['job'],points=points,pool=pool,kept=kept,dropped=dropped,pool_verdict=pool_verdict,limits='Selection means, not G0 promotion. All point CIs disclosed; mean-positive uncertain points remain exploratory. Parent pool schedule uses its changed-load mean to dispatch each T, then the new whole-graph batch prices the composition.')

def apply(source,selection):
 sp=importlib.util.spec_from_file_location('ng_parent_exact_patch',P/'patch.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);out=mod.apply(source,selection['points']);pool={int(t):a for t,a in selection.get('pool',{}).items()}
 if not pool:return out
 out+='\n'+(P/'pool_tma.py.inc').read_text()+'\n'+(P/'pool.py.inc').read_text()+'\nNG_PARENT_POOL='+repr(pool)+'''
_ng_pool_base_project=_best7_project
_ng_pool_base_prepare=prepare
def _best7_project(x,role,state,plan,out_dtype=torch.bfloat16):
    if role=='pool_router' and x.shape[0] in NG_PARENT_POOL:
        bank,pool=state['ng_parent_pool']
        return pp_pool_project(x,bank,pool,NG_PARENT_POOL[x.shape[0]])
    return _ng_pool_base_project(x,role,state,plan,out_dtype)
def prepare(weights):
    global _best7_project
    selected=_best7_project;_best7_project=_ng_pool_base_project
    try:state=_ng_pool_base_prepare(weights)
    finally:_best7_project=selected
    bank=pp_prepare(globals(),weights,state,{(t,'pool_router','P') for t,a in NG_PARENT_POOL.items() if a in PP_POOL_TMA})
    pool=pp_pool_prepare(globals(),state) if any(a in PP_POOL_NF for a in NG_PARENT_POOL.values()) else None
    if any(PP_POOL_TMA.get(a,(0,))[0]==128 for a in NG_PARENT_POOL.values()):
        w=state['pool_router'];bank['tiled']['pool_router',128]=w.reshape(72,64,64,128).permute(0,2,1,3).contiguous().view(-1,128)
    state['ng_parent_pool']=(bank,pool)
    for t,a in NG_PARENT_POOL.items():pp_pool_project(torch.zeros((t,8192),device=weights['qkv_gate'].device,dtype=torch.bfloat16),bank,pool,a)
    torch.cuda.synchronize()
    return state
'''
 compile(out,'<native selected parent portfolio>','exec');return out

if __name__=="__main__":
 import sys,hashlib
 source=Path(sys.argv[1]).read_text();selection=json.loads(Path(sys.argv[3]).read_text());out=apply(source,selection);Path(sys.argv[2]).write_text(out);print(hashlib.sha256(out.encode()).hexdigest())
