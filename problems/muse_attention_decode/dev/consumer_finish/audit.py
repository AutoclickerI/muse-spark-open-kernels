"""Prove the oracle retains source outside the declared consumer-input work."""
import ast,hashlib,json,re
from pathlib import Path
import kernels
H=Path(__file__).resolve().parent

def original():
 out={};keys=set(kernels.sources())
 for n in ast.parse(kernels.base()).body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if getattr(t,'id','') in keys:out[t.id]=ast.literal_eval(n.value)
 return out

def blank(s,names):
 for name in names:
  if name in s:s=kernels.body(s,name,'')
 s=re.sub(r'DI uint16_t cf_ready\([^\n]+\n','',s)
 return re.sub(r'\s+','',s)

def main():
 old=original();new=kernels.sources();rows=[]
 for key,changed in new.items():
  for sub,target in (changed.items() if isinstance(changed,dict) else [(key,changed)]):
   source=old[key][sub] if isinstance(changed,dict) else old[key]
   if key in ('ATTN_PORT_CUDA_SRC','PARTIAL_CUDA'):
    a=source.index('  constexpr int QR =');b=source.index('  WarpState<L> st;',a)
    aa=target.index('  const float ksc=',a-2);bb=target.index('  WarpState<L> st;',aa)
    assert source[:a].rstrip()==target[:aa].rstrip() and source[b:]==target[bb:]
    kept='Exact source prefix/suffix outside main input prologue, including full history issue loop, tensor math, softmax and merge.'
   elif key=='VQ8_CUDA':
    anchor='  asm volatile("griddepcontrol.wait;" ::: "memory");'
    want=source.replace(anchor,'  // Oracle input is ready outside the timed call.',1)
    assert blank(want,['DI void prologue('])==blank(target,['DI void prologue('])
    kept='All source except head prologue and first/main input PDL wait; combine wait is retained.'
   else:
    names=['DI void prep(','DI void prep_decode(','DI void prep_verify(','DI void prep_new(']
    assert blank(source,names)==blank(target,names)
    kept='All source except input-preparation bodies and new ready-byte helper; math, tile loops, virtual-key attention and DSM merge unchanged.'
    if key=='wr_KERNELS':
     at=target.index('DI void prep_verify(');end=target.index('DI void prep_new(',at)
     assert '*(uint4*)(KN' not in target[at:end] and '*(uint4*)(VN' not in target[at:end]
   rows.append(dict(name=key+'_'+sub,original_sha256=hashlib.sha256(source.encode()).hexdigest(),oracle_sha256=hashlib.sha256(target.encode()).hexdigest(),retained=kept))
 (H/'audit.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS',len(rows),'source-boundary proofs, including RAW2 live-storage guard')
if __name__=='__main__':main()
