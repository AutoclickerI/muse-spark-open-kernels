"""Pin production-capable reset cells from immutable queued wrappers."""
import ast,base64,hashlib,json,lzma,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]

def segment(source,node):
 start=min([node.lineno]+[n.lineno for n in getattr(node,'decorator_list',[])])
 return ''.join(source.splitlines(True)[start-1:node.end_lineno])

def values(s):
 out={}
 for n in ast.parse(s).body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name):
     try:out[t.id]=ast.literal_eval(n.value)
     except (ValueError,TypeError):pass
 return out

def body(s):
 n=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
 return lzma.decompress(base64.b85decode(ast.literal_eval(n.args[0]))).decode()

def driver(s,prefix):
 nodes=[]
 for n in ast.parse(s).body:
  if isinstance(n,ast.FunctionDef) and n.name.startswith(prefix) and n.name[len(prefix):] in ('module','plan','attention','bind','experts'):nodes.append(n)
  elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith(prefix) for t in n.targets):
   try:ast.literal_eval(n.value);nodes.append(n)
   except (ValueError,TypeError):pass
 return '\n\n'.join(segment(s,n) for n in nodes)

def delta(old,new):
 old_map={};out=[];occ={}
 def key(n):
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)):return type(n).__name__,n.name
  if isinstance(n,ast.Assign):return 'Assign',ast.unparse(n.targets)
  return type(n).__name__,ast.dump(n)
 for n in ast.parse(old).body:
  k=key(n);i=occ.get(k,0);occ[k]=i+1;old_map[k+(i,)]=n
 occ={}
 for n in ast.parse(new).body:
  k=key(n);i=occ.get(k,0);occ[k]=i+1;o=old_map.get(k+(i,))
  if o is not None and ast.dump(o)==ast.dump(n):continue
  out.append(dict(kind=k[0],name=k[1],ordinal=i,old=segment(old,o) if o else None,new=segment(new,n)))
 return out

def main():
 p=H/'generated/payloads';p.mkdir(parents=True,exist_ok=True)
 for entry in json.loads((H/'reset_inventory.json').read_text()):
  wrapper=(R/entry['file']).read_text();assert hashlib.sha256(wrapper.encode()).hexdigest()==entry['sha256'],entry['file']
  for n in ast.parse(wrapper).body:
   if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in n.targets):
    for name,digest,text in ast.literal_eval(n.value):(p/(name+'.py')).write_text(text)
 s14=(R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text();cells={}
 glob=(p/'global_attention_pair.py').read_text();cells['global']=dict(code=driver(glob,'cs_'))
 board=(R/'problems/muse_attention_decode/dev/cluster_sliding/out/p5_dsm_board.py').read_text()
 code=driver(board,'cc_');v=values(board);cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',v['cc_CUDA'],re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
 code=re.sub(r'^cc_CPP=.*$', 'cc_CPP = '+repr(cpp).replace('\\','\\\\'),code,flags=re.M) if False else code
 # The queued board recomputes CPP with re at import; retain its exact computed declarations.
 ls=code.splitlines(True)
 for n in reversed(ast.parse(code).body):
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='cc_CPP':ls[n.lineno-1:n.end_lineno]=['cc_CPP = '+repr(cpp)+'\n']
 cells['dsm']=dict(code=''.join(ls))
 early=body((p/'early_history.py').read_text());hist={}
 for n in ast.parse(early).body:
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and getattr(n.value.func,'id','')=='exec' and len(n.value.args)>1:
   if isinstance(n.value.args[0],ast.Constant) and isinstance(n.value.args[1],ast.Name) and n.value.args[1].id.startswith('HIST_'):
    hist[n.value.args[1].id]=n.value.args[0].value
 cells['early']=dict(early=hist['HIST_EARLY'],late=hist['HIST_LATE'])
 soft=values((p/'softmax_mix_lab.py').read_text())['SOURCES']
 cells['soft']={name:values(src)['VQ8_CUDA'] for name,src in soft.items() if name not in ('A','ref')}
 epi=(p/'moe_nt1_epilogue.py').read_text();ev=values(epi)
 cells['epi']={}
 for name,prefix in ev['EPI_ARMS'].items():
  if name=='EPBASE':continue
  code=driver(epi,prefix+'_')
  # Include the exact CUDA/CPP/flag/cache assignments preceding its driver functions.
  code='\n'.join(k+' = '+repr(ev[k]) for k in (prefix.upper()+'_CUDA',prefix.upper()+'_CPP',prefix.upper()+'_FLAGS_BUILT','_'+prefix.upper()+'_MODULE'))+'\n'+code
  cells['epi'][name]=dict(code=code,prefix=prefix)
 heavy=(p/'cluster_schedule_payload.py').read_text();hv=values(heavy)
 # Include only the preparation/route/launch closure, not synthetic fixtures or profiling.
 defs={n.name:n for n in ast.parse(heavy).body if isinstance(n,ast.FunctionDef)}
 decl={t.id:n for n in ast.parse(heavy).body if isinstance(n,ast.Assign) for target in n.targets for t in ast.walk(target) if isinstance(t,ast.Name)}
 picked=set();todo=['oc_plan','m128_hh_issue_nt_prepare'];external={'nf_module','m128_hh_pm_module','NF_OPTS','load_inline','torch'}
 while todo:
  name=todo.pop()
  if name in picked or name in external:continue
  node=defs.get(name,decl.get(name))
  if node is None:continue
  picked.add(name)
  todo += [n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)]
 seq=[]
 for n in ast.parse(heavy).body:
  names=[n.name] if isinstance(n,ast.FunctionDef) else [t.id for target in n.targets for t in ast.walk(target) if isinstance(t,ast.Name)] if isinstance(n,ast.Assign) else []
  if any(name in picked for name in names):seq.append(segment(heavy,n))
 cells['heavy']=dict(code='\n\n'.join(seq),names=sorted(picked),cuda=hv['OP_CUDA'],cpp=hv['OP_CPP'],arms=list(hv['OP_NAMES']))
 bd=values(body((p/'lab_boundary_batch.py').read_text()))
 cells['boundary']={name:delta(s14,src) for name,src in bd['BD_SOURCES'].items()}
 pa=values(body(bd['BD_CONFIRM_SOURCE']))
 # Confirmation already pins complete S14 variants. Exclude the measured, grossly losing
 # all-register-tile F control; B/M/D are production-capable distinct recipes.
 print('PA bindings',[(k,len(v) if isinstance(v,str) else type(v).__name__) for k,v in pa.items() if k.startswith('PA_')])
 cells['pdl']={name:delta(s14,pa[key]) for name,key in [('B','PA_HINT'),('M','PA_META'),('D','PA_D1')]}
 result=dict(base_sha256=hashlib.sha256((R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_bytes()).hexdigest(),cells=cells)
 (H/'cells.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
 print('pinned categories',list(cells),'bytes',(H/'cells.json').stat().st_size)
if __name__=='__main__':main()
