"""Prove the lab's selected-only setup matches S11's full prepare on scored paths."""
import argparse,hashlib,json,runpy,sys,time,types
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev'),str(ROOT/'problems/muse_decode_layer')]
import local_eval
import reference
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');a=ap.parse_args()
mode='selected_forced' if a.forced else 'selected_normal'
ns=runpy.run_path(str(HERE/'lab_timeline.py'))
host=local_eval.load_module('timeline_setup_control',ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py')
host._ap_fa3_scheduler_metadata=lambda *a,**k:None
if a.forced:
 import verify_best
 verify_best.install(host,False)
weights=reference.generate_weights(5353,device='cuda')
full=host.prepare(weights)
t0=time.monotonic();selected=ns['tl_prepare_selected'](host,weights);setup_s=time.monotonic()-t0
# Exact dtype/layout/value checks on every dense/glue tensor and every timed bank.
checked=[]
def compare(path,x,y):
 if isinstance(x,torch.Tensor):
  assert x.dtype==y.dtype and x.shape==y.shape and x.stride()==y.stride(),path
  assert torch.equal(x,y),path
  checked.append(path)
 elif isinstance(x,dict):
  for k,v in x.items():
   if k in y:compare(path+'.'+str(k),v,y[k])
 elif isinstance(x,(tuple,list)):
  for i,(u,v) in enumerate(zip(x,y)):compare(path+'.'+str(i),u,v)
for key in ('qkv_gate','qkv3','gate_rows','o','pool_router','unpooler','unpooler_slice','router_bias',
            'alpha1','alpha2','beta1','beta2','pre_ffn_w','pre_expert_w','post_ffn_w','next_pre_attn_w',
            's4_nt1_bank','s5_hh_base','moe_bank'):
 compare(key,full[key],selected[key])
assert full['best7_plans']==selected['best7_plans']
assert full['vq8_rows']==selected['vq8_rows']
for key,p in full['best7_plans'].items():
 if key=='other':continue
 b,q,w=key;t=b*q
 for name in ('qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler'):
  if p['projection'].get(name,'auto')=='auto':
   assert full['proj_plan'][(t,name)]==selected['proj_plan'][(t,name)],(t,name)
rows=[]
for b,q,w in ns['TL_CASES']:
 for seed in (172903,180822,188741):
  d=host._ap_generate_input(batch=b,ctx=80000,layer='sliding' if w else 'full',q_len=q,
                           spread='fixed' if b==1 else 'agentic',dist='soft',page=1,seed=seed,device='cuda')
  outputs=[];graphs=[]
  for state in (full,selected):
   for _ in range(2):host.kernel_fn(d,state)
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=host.kernel_fn(d,state)
   g.replay();torch.cuda.synchronize();outputs.append(out);graphs.append(g)
  x,y=outputs
  assert torch.equal(x[2],y[2]),(b,q,w,seed,'ids')
  rel=[float((u.float()-v.float()).norm()/u.float().norm().clamp_min(1e-12)) for u,v in zip(x[:2],y[:2])]
  assert max(rel)<.001,(b,q,w,seed,rel)
  rows.append(dict(case=[b,q,w],seed=seed,rel=rel))
  del d,outputs,graphs,x,y,out,g
buf=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
ns['lab'].__globals__.update(TL_EXISTING_HOST=(host,selected),lab_cold_l2=lambda:buf.view(torch.int64).sum())
lines=ns['lab'](time.monotonic()+260,weights)
assert len(ns['lab'].__globals__['TL_RESULTS'])==12
assert not any(l.startswith(('FAIL','INCOMPLETE')) for l in lines)
result=dict(payload_sha256=hashlib.sha256((HERE/'lab_timeline.py').read_bytes()).hexdigest(),setup_s=setup_s,
            exact_tensors=checked,fixtures=rows,lines=[l for l in lines if not l.startswith('TL_XZ85')])
(HERE/(mode+'_local.json')).write_text(json.dumps(result,indent=2)+'\n')
print('PASS',mode,'selected setup',setup_s,'exact tensors',len(checked),'whole-layer fixtures',len(rows),'timeline shapes 12',flush=True)
