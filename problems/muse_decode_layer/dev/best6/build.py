"""Embed the unchanged prefill expert adapters in the pinned glue source."""
from pathlib import Path
import ast,hashlib,copy
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best5_glue.py';PREFILL=ROOT/'problems/muse_prefill_layer/cand_mid_sweep.py';OUT=BASE.with_name('cand_r3_best6.py')
s=BASE.read_text();assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='9912056970eef92aea847fd1cb68e7426ce73df738eef739e2f63ca1837fb714'
p=PREFILL.read_text();assert hashlib.sha256(PREFILL.read_bytes()).hexdigest()=='1de6c32c7a5337dc539a5b6feb24a479bed217cb794a92a32760f9fcda82e075'
t=ast.parse(p)
defs={n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assigns={x.id:n for n in t.body if isinstance(n,ast.Assign) for x in n.targets if isinstance(x,ast.Name)}
roots={'_raw_experts_tiled','_run_canonical_dg','_experts_humming_masked','_raw_repack','_encode_blob_fast','_encode_sf_fast','repack_v6','_Experts','_RawModule','_grouped_module'}
needed=set();todo=list(roots)
while todo:
 name=todo.pop()
 if name in needed:continue
 node=defs.get(name,assigns.get(name))
 if node is None:continue
 needed.add(name)
 for n in ast.walk(node):
  if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and (n.id in defs or n.id in assigns) and n.id not in needed:todo.append(n.id)
# Imported third-party names are deferred until optional prepare setup.
third={'GemmType','WeightScale2Type','HummingKernel','HummingMethod','ModeloptNvfp4WeightSchema','get_heuristics_config','humming_ops'}
renamed=needed|third
class Prefix(ast.NodeTransformer):
 def __init__(self):self.scopes=[]
 def visit_Name(self,n):
  if n.id in renamed and not any(n.id in scope for scope in reversed(self.scopes)):
   n.id='_pf'+n.id if n.id.startswith('_') else '_pf_'+n.id
  return n
 def visit_FunctionDef(self,n):
  if n.name in renamed:n.name='_pf'+n.name if n.name.startswith('_') else '_pf_'+n.name
  local={a.arg for a in (*n.args.posonlyargs,*n.args.args,*n.args.kwonlyargs)}
  if n.args.vararg:local.add(n.args.vararg.arg)
  if n.args.kwarg:local.add(n.args.kwarg.arg)
  for x in ast.walk(n):
   if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store):local.add(x.id)
   if isinstance(x,(ast.Import,ast.ImportFrom)):
    local.update(v.asname or v.name.split('.')[0] for v in x.names)
  for x in ast.walk(n):
   if isinstance(x,ast.Global):local.difference_update(x.names)
  self.scopes.append(local);result=self.generic_visit(n);self.scopes.pop();return result
 def visit_ClassDef(self,n):
  if n.name in renamed:n.name='_pf'+n.name if n.name.startswith('_') else '_pf_'+n.name
  return self.generic_visit(n)
 def visit_Global(self,n):
  n.names=['_pf'+x if x.startswith('_') else '_pf_'+x for x in n.names];return n

nodes=[]
for n in t.body:
 if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in needed:nodes.append(copy.deepcopy(n))
 elif isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in needed for x in n.targets):nodes.append(copy.deepcopy(n))
module=Prefix().visit(ast.Module(nodes,type_ignores=[]));ast.fix_missing_locations(module)
embedded='\n\n# Exact prefill adapter ASTs; all global names isolated from the decode layer.\nimport json,os\n'+ast.unparse(module)+'\n'
# Launch optional native compilation concurrently with the existing base prepare.
# CPU native compilation may overlap weight repacking. GPU adapter setup must
# finish before any CUDA graph capture; concurrent GPU setup is not permitted by
# CUDA's global capture mode.
a=s.index('    import time\n    global _BEST5_DEADLINE',s.index('def prepare('));b=s.index("    phases={'weights':",a)
s=s[:a]+'''    import time
    global _BEST5_DEADLINE,_BEST6_GROUPED_ERROR
    _BEST6_GROUPED_ERROR=None
    started=time.monotonic();_BEST5_DEADLINE=started+235.
    from concurrent.futures import ThreadPoolExecutor
    device=weights['qkv_gate'].device
    with ThreadPoolExecutor(max_workers=7) as native_pool:
        grouped_future=native_pool.submit(_pf_grouped_module)
        nt2_future=native_pool.submit(_best5_nt2_module)
        attention_futures=[]
        if _attn_port_hopper(device):
            attention_futures=[native_pool.submit(_attn_port_ext),
                native_pool.submit(_attn_port_xext,'group'),native_pool.submit(_attn_port_xext,'warp'),
                native_pool.submit(_attn_port_p8ext)]
        state=_prepare_weights(weights)
        try:grouped_future.result()
        except Exception as exc:_BEST6_GROUPED_ERROR=_best6_reason(exc)
        state['moe_nt2_mod']=nt2_future.result()
        try:state['moe6_adapters']=_best6_build_adapters(weights)
        except Exception as exc:state['moe6_adapters']={'available':[],'rejected':{name:_best6_reason(exc) for name in ('raw17','humming_masked','dg_w8')}}
        for future in attention_futures:
            try:future.result()
            except Exception as exc:print('BEST6_OPTIONAL_ATTN_COMPILE',_best6_reason(exc),flush=True)
'''+s[b:]
# Keep best5's V9 NT comparison for a measured default, then prioritize the
# unified portfolio before dense/attention/geometry tuning consumes the budget.
s=s.replace('    _best5_tune_moe(state,weights,flush,device)',
            '    _best5_tune_moe(state,weights,flush,device)\n    _best6_tune(state,weights,flush,device)',1)
s=s.replace('    moe=experts_v9(x,wts,lids,state)','    moe=_best6_experts(pr,x,wts,lids,state)',1)
s=s.replace("    for m,row in state['moe_nt_rows'].items():print(f'BEST5_NT M={m} {row}',flush=True)","    for m,row in state['moe_nt_rows'].items():print(f'BEST5_NT M={m} {row}',flush=True)\n    _best6_summary(state)",1)
s=s.replace('"""Best5 with self-contained PDL residual/norm and routing glue.','"""Best6: glue base with FP32-gated prefill expert adapters at M128/256.')
s+=embedded+'\n'+(HERE/'portfolio.py.inc').read_text()
OUT.write_text(s)
manifest=dict(base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),prefill_sha256=hashlib.sha256(PREFILL.read_bytes()).hexdigest(),
              sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),embedded_definitions=sorted(needed),third_party_names=sorted(third))
(HERE/'build_manifest.json').write_text(__import__('json').dumps(manifest,indent=2)+'\n')
print(OUT,manifest['sha256'],len(needed),'prefill definitions')
