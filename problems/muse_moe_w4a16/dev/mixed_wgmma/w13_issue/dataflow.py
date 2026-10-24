"""Classify steady-loop static SASS by Q/SF register provenance, with explicit limits."""
import re,json,collections
from pathlib import Path
from count import parse
H=Path(__file__).resolve().parent;P=H.parent

def classify(ins):
 tags={};count=collections.Counter()
 for pc,op,arg,pred in ins:
  fields=arg.split(',');first=fields[0];dest=re.match(r'R(\d+)',first.strip());dr=int(dest[1]) if dest else None
  src=set()
  for n in re.findall(r'\bR(\d+)',','.join(fields[1:])):src|=tags.get(int(n),set())
  value=set();write=dr is not None
  if pred=='@!PT':kind='predicated_off_fence_scaffold';write=False
  elif op.startswith(('HGMMA','WARPGROUP')):kind='WGMMA_issue_fence_wait';write=False
  elif op.startswith(('LDGSTS',)):
   kind='async_weight_activation_load';write=False
  elif op.startswith(('LD.E','LDS')):
   if '.64' in op:kind='Q_shared_load';value={'Q'}
   elif 'SF' in src:kind='scale_lookup_load';value={'SF'}
   else:kind='SF_shared_load';value={'SF'}
  elif op.startswith('HMUL2'):kind='scale_multiply';value={'A'}
  elif op.startswith(('LOP3','SHF','SHFL','IMAD','LEA','PRMT','IADD','VIADD')) and src:
   if 'Q' in src:kind='FP4_unpack';value={'Q'}
   elif 'SF' in src:kind='scale_decode';value={'SF'}
   else:kind='address_integer'
  elif op.startswith(('IADD','IMAD','LEA','UIADD','VIADD','ULEA','USHF','ULOP','UMOV','SHF','LOP','PRMT')):kind='address_integer'
  elif op.startswith(('LDC','S2R')):kind='address_constant_state'
  else:kind='control_barrier_other'
  count[kind]+=1
  if write:
   tags[dr]=value
   if op.startswith(('LDS.64','LD.E.64','LDC.64')):tags[dr+1]=value
 return dict(count)

def main():
 rows=[]
 for name in ('raw','recoded'):
  ins=[]
  for line in (H/(name+'_loop.txt')).read_text().splitlines():
   m=re.match(r'([0-9a-f]+)\s+(@!?\w+)?\s*([A-Za-z][\w.]*)\s+(.*)',line)
   assert m,line;ins.append((int(m[1],16),m[3],m[4],m[2] or ''))
  c=classify(ins);rows.append(dict(name=name,total=len(ins),per16={k:v/4 for k,v in c.items()}))
 (H/'dataflow_counts.json').write_text(json.dumps(rows,indent=2)+'\n')
 for row in rows:print(row)

if __name__=='__main__':main()
