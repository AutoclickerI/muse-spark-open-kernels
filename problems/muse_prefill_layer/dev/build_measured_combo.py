"""Build one complete candidate from completed per-stage H200 measurements.

This script is development orchestration, never part of a submitted kernel.
Expert families retain their checked shape dispatch. Attention/dense dispatch uses
three broad token-size buckets, with all keys and original precisions preserved.
"""
import argparse,ast,datetime,json,re,statistics
from collections import defaultdict
from pathlib import Path
P=Path(__file__).resolve().parents[1];ROOT=P.parents[1]
EXPERTS=(2,3,6,7,8,10,11,12,14,15,16,17)
REQUIRED=(2,3,4,6,7,8,9,10,11,12,13,14,15,16,17)
LOADS=('turn 1','turn 4','first turn')

def receipts():
 out={}
 bad=failed_score_hashes()
 for f in (ROOT/'runs/muse_prefill_layer').glob('*_check_*.json'):
  d=json.loads(f.read_text());source=f.with_suffix('.py')
  if not source.exists():continue
  for n in REQUIRED:
   c=P/f'cand_v{n}.py'
   if c.exists() and source.read_bytes()==c.read_bytes():
    if __import__('hashlib').sha256(source.read_bytes()).hexdigest() in bad:
     d['result']['flags']=['failed_scored_rerun']
    out[n]=(d,f);break
 return out

def failed_score_hashes():
 import hashlib
 bad=set()
 for f in (ROOT/'runs/muse_prefill_layer').glob('*_submit_*.json'):
  d=json.loads(f.read_text());r=d.get('result',{})
  if r.get('passed') is False or r.get('flags'):
   source=f.with_suffix('.py')
   if source.exists():bad.add(hashlib.sha256(source.read_bytes()).hexdigest())
 return bad

def passed(d):
 r=d.get('result',{});return r.get('passed') and not r.get('flags') and r.get('speedup')

def prof(d):
 out={}
 for x in d.get('result',{}).get('records',[]):
  if x.get('event')!='profile':continue
  c=x['case'];v=defaultdict(float)
  for line in x['report'].splitlines():
   m=re.match(r'(.+?)\s+([\d.]+) µs',line)
   if not m:continue
   name,us=m.group(1),float(m.group(2))
   if 'flash::' in name or '_gather_cache' in name:v['attention']+=us
   if 'deep_gemm::' in name:v['dg']+=us
   if name.startswith(('nvjet','ampere')):v['cublas']+=us
   if any(z in name for z in ('Marlin','deep_gemm::','humming','moe_gemm_kernel','_dequant_kernel','_swiglu','_combine_kernel','_gather_norm','_pre_expert_norm','_post_expert_norm','act_and_mul','combine_tiled','gather_tiled','tile_offsets')):v['expert']+=us
  key=(c['load'],bool(c['layer']=='sliding'))
  if key not in out:out[key]=defaultdict(float)
  for k,z in v.items():out[key][k]+=z*c['weight']
  out[key]['weight']+=c['weight']
 for v in out.values():
  for k in tuple(v):
   if k!='weight':v[k]/=v['weight']
 return out

def plan(rs):
 eligible={n:d['result']['speedup'] for n,(d,f) in rs.items() if n in EXPERTS and passed(d)}
 best=max(eligible,key=eligible.get)
 at={};dense={};expert={};evidence=[]
 control=prof(rs[2][0]);p3=prof(rs[3][0])
 for load in LOADS:
  ec={}
  for n in eligible:
   pr=prof(rs[n][0]);keys=[k for k in pr if k[0]==load]
   if keys:
    weight=sum(pr[k]['weight'] for k in keys)
    ec[n]=sum(pr[k]['expert']*pr[k]['weight'] for k in keys)/weight
  chosen=min(ec,key=ec.get)
  if ec[chosen]>0.97*ec[best]:chosen=best
  expert[load]=chosen
  evidence.append(dict(stage='experts including listed glue',load=load,us=ec,selected=chosen,margin='Retain overall best family unless another improves its profiled expert stage by at least3%.'))
  for sliding in (False,True):
   key=(load,sliding)
   if key not in control:continue
   # Use the same v2 glue/expert control for the attention experiments.
   default=control[key]['attention'];choices={'paged':default}
   for n,name in ((4,'contiguous'),(9,'unpacked')):
    if n in rs and passed(rs[n][0]):
     pr=prof(rs[n][0]);
     if key in pr:choices[name]=pr[key]['attention']
   picked=min(choices,key=choices.get)
   # Keep measured baseline when the isolated gain is small/noisy.
   if choices[picked]>default*0.97 or default-choices[picked]<3:picked='paged'
   at[f'{load}:{int(sliding)}']=picked
   evidence.append(dict(stage='attention',load=load,sliding=sliding,us=choices,selected=picked))
  # Only the native dense trial differs from v2's cuBLAS control. On large T,
  # subtract control grouped expert GEMM time to estimate the native dense sum.
  # This estimate is for dispatch; reports must not call it an exact stage split.
  keys=[k for k in control if k[0]==load]
  mode='cublas'
  if 13 in rs and passed(rs[13][0]):
   native=prof(rs[13][0]); comparisons=[]
   for key in keys:
    if key not in native:continue
    old=control[key]['cublas'];new=native[key]['dg']-control[key]['dg']+native[key]['cublas']
    if old>0 and new>0:comparisons.append((old,new))
   if comparisons and sum(b for a,b in comparisons)<0.95*sum(a for a,b in comparisons):mode='deepgemm'
   evidence.append(dict(stage='dense',load=load,control_native_us=comparisons,selected=mode,method='Native DG minus unchanged grouped-expert DG; includes router in both sides; inferred dispatch estimate.'))
  dense[load]=mode
 return dict(base=best,base_job=rs[best][0]['id'],base_speedup=eligible[best],experts=expert,attention=at,dense=dense,evidence=evidence)

def expert_namespace(n):
 """Static private definitions, without importing or patching another candidate module."""
 tree=ast.parse((P/f'cand_v{n}.py').read_text())
 drop={'kernel_fn','_attention','_qk_prep_kernel','_attn_fwd_kernel','_norm_gate_kernel','_boundary_kernel','_routing_kernel','_bench_w4a16','_marlin_bank','_experts_marlin'}
 tree.body=[x for x in tree.body if not isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) or x.name not in drop]
 minimal='''
st={
 'pre_expert':(1+weights['pre_expert_norm'].float()).contiguous(),
 'post_expert':(1+weights['post_expert_norm'].float()).contiguous(),
 'w13':weights['w13'].contiguous(), 'w13_sf':weights['w13_sf'].view(torch.uint8).contiguous(),
 'w2':weights['w2'].contiguous(), 'w2_sf':weights['w2_sf'].view(torch.uint8).contiguous(),
 'w13_gs':weights['w13_gs'].float().contiguous(), 'w2_gs':weights['w2_gs'].float().contiguous(), 'bank':None,
}
st['post_expert_bf16']=st['post_expert'].to(torch.bfloat16)
return st
'''
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and (node.name=='prepare_base' or (n in (2,10) and node.name=='prepare')):
   node.body=ast.parse(minimal).body
 names=set()
 for node in tree.body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):names.add(node.name)
  elif isinstance(node,ast.Assign):
   for target in node.targets:
    names.update(x.id for x in ast.walk(target) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store))
 mapping={name:f'e{n}_{name}' for name in names}
 class Rename(ast.NodeTransformer):
  def visit_Name(self,node):node.id=mapping.get(node.id,node.id);return node
  def visit_arg(self,node):node.arg=mapping.get(node.arg,node.arg);return self.generic_visit(node)
  def visit_keyword(self,node):node.arg=mapping.get(node.arg,node.arg);return self.generic_visit(node)
  def visit_Global(self,node):node.names=[mapping.get(x,x) for x in node.names];return node
  def visit_FunctionDef(self,node):node.name=mapping.get(node.name,node.name);return self.generic_visit(node)
  def visit_ClassDef(self,node):node.name=mapping.get(node.name,node.name);return self.generic_visit(node)
 tree=Rename().visit(tree);ast.fix_missing_locations(tree)
 return ast.unparse(tree)

def expert_call(n):
 prefix=f'e{n}_'
 args="pooled,topk_w,slot_e,slot_pos,counts,bank"
 if n in (2,10):return f'{prefix}_experts_dequant({args})'
 if n in (3,7,15,17):return f'{prefix}experts_tiled({args})'
 if n in (6,14):return f'{prefix}_experts_humming(pooled,topk_w,slot_e,bank)'
 if n==8:return f'({prefix}_experts_humming_masked({args}) if _load_bucket(T)=="turn 1" else {prefix}_experts_dequant({args}))'
 if n in (11,16):return f'({prefix}experts_tiled({args}) if _load_bucket(T)!="first turn" else {prefix}_experts_dequant({args}))'
 if n==12:return f'({prefix}experts_tiled({args}) if _load_bucket(T)!="first turn" else {prefix}experts_tiled256({args}))'
 raise ValueError(n)

def build(pl,output):
 s=(P/'cand_v2.py').read_text()
 i=s.index('    try:\n        st["bank"] = _marlin_bank(weights)');j=s.index('    torch.cuda.synchronize()',i)
 s=s[:i]+'    st["bank"] = None\n'+s[j:]
 s=s.replace('def prepare(weights):','def _prepare_dense_state(weights):')
 i=s.index('    if T >= BIG_TOKENS or st["bank"] is None:');j=s.index('    if T <= 8:',i)
 s=s[:i]+'    moe = _experts_selected(pooled,topk_w,slot_e,slot_pos,counts,st)\n'+s[j:]
 families=sorted(set(pl['experts'].values()))
 for n in families:s+='\n\n'+expert_namespace(n)
 s+='\n\n_EXPERT_MODES = '+repr(pl['experts'])+'\n'
 s+='\n\ndef prepare(weights):\n'
 if 'deepgemm' in pl['dense'].values():s+='    import deep_gemm\n    deep_gemm.use_deterministic_algorithms(True)\n'
 s+='    st=_prepare_dense_state(weights)\n    st["expert_states"]={}\n'
 for n in families:s+=f'    st["expert_states"][{n}]=e{n}_prepare(weights)\n'
 s+='    return st\n\n'
 s+='def _experts_selected(pooled,topk_w,slot_e,slot_pos,counts,st):\n    T=pooled.shape[0]\n    family=_EXPERT_MODES[_load_bucket(T)]\n    bank=st["expert_states"][family]\n'
 for n in families:s+=f'    if family=={n}:return {expert_call(n)}\n'
 s+='    raise RuntimeError("Unknown expert family")\n'
 # Public timed call invokes selected components. Preserve the original paged FA3 function.
 s=s.replace('attn = _attention(data, q8, st)','attn = _attention_selected(data, q8, st)')
 for x,w in [('x','qkv'),('o_in','o'),('h','pooler'),('moe','unpooler')]:
  s=s.replace(f'torch.mm({x}, st["{w}"].t())',f'_dense_selected({x}, st["{w}"])')
 att=(P/'dev/attention_dense.py').read_text()
 att=att.replace('def _attention_contiguous(data, q8, st):','def _attention_contiguous(data, q8, st, unpacked=False):')
 att=att.replace('window_size=(W-1,0) if W else (-1,-1), **descale)', 'window_size=(W-1,0) if W else (-1,-1),\n                pack_gqa=False if unpacked and B*triton.cdiv(L,128)*HQ >= 132 else None,\n                num_splits=1 if unpacked and B*triton.cdiv(L,128)*HQ >= 132 else 0, **descale)')
 s+='\n\n'+att
 s+='\n\n_ATTENTION_MODES = '+repr(pl['attention'])+'\n_DENSE_MODES = '+repr(pl['dense'])+'\n'
 s+='''

def _load_bucket(t):
    return 'turn 1' if t < 2048 else ('turn 4' if t < 8192 else 'first turn')

def _attention_selected(data,q8,st):
    cfg=data['config']
    mode=_ATTENTION_MODES.get(_load_bucket(cfg['tokens'])+':'+str(int(bool(cfg['window']))),'paged')
    if mode=='paged':return _attention(data,q8,st)
    return _attention_contiguous(data,q8,st,unpacked=mode=='unpacked')

def _dense_selected(x,w):
    if IS_HOPPER and _DENSE_MODES[_load_bucket(x.shape[0])]=='deepgemm':
        import deep_gemm
        y=torch.empty((x.shape[0],w.shape[0]),dtype=torch.bfloat16,device=x.device)
        deep_gemm.bf16_gemm_nt(x,w,y)
        return y
    return torch.mm(x,w.t())
'''
 s=s.replace('Muse Spark Open prefill layer:', 'Measured complete component integration:')
 # Keep only the central dense/attention/routing glue and private expert definitions.
 dead={'_marlin_bank','_bench_w4a16','_experts_marlin','_experts_dequant','_offsets_kernel','_gather_norm_kernel','_e2m1','_dequant_kernel','_swiglu_kernel','_combine_kernel','_pre_expert_norm_kernel'}
 tree=ast.parse(s)
 tree.body=[x for x in tree.body if not isinstance(x,ast.FunctionDef) or x.name not in dead]
 for node in tree.body:
  if isinstance(node,ast.FunctionDef):
   node.body=[x for x in node.body if not (isinstance(x,ast.If) and isinstance(x.test,ast.Constant) and x.test.value is False)]
 s='# okc: problem=muse_prefill_layer\n'+ast.unparse(tree)+'\n'
 if output.exists():raise RuntimeError('Refusing to overwrite '+str(output))
 output.write_text(s)
 return output

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allow-incomplete',action='store_true');ap.add_argument('--output',default='cand_v18.py');ap.add_argument('--plan-only',action='store_true');args=ap.parse_args()
 rs=receipts();missing=sorted(set(REQUIRED)-rs.keys())
 if missing and not args.allow_incomplete:raise SystemExit('Pending component receipts: '+str(missing))
 pl=plan(rs);pl['missing']=missing;pl['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (P/'dev/combo_plan.json').write_text(json.dumps(pl,indent=2));print(json.dumps(pl,indent=2))
 if not args.plan_only:build(pl,P/args.output)
if __name__=='__main__':main()
