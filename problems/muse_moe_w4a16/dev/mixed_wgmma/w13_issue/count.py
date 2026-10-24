"""Count the actual W13 K128 loop, and normalize warp-issued instructions per16 weights."""
import re,json,collections
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent

def parse(s):
 out=[]
 for line in s.splitlines():
  m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(?:(@!?[UP]\w*)\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;',line)
  if m:out.append((int(m[1],16),m[3],m[4],m[2] or ''))
 return out

def main():
 rows=[]
 for label,path,marker in [('raw',P/'diagnostics/w2fix.sass','mwfx5fusedILi32ELi128ELi4ELi1ELb1ELi1'),('raw_pair',P/'recode/native.sass','mwrc5fusedILi32ELi128ELi4ELi1ELb1ELi1ELb0'),('recoded',P/'recode/native.sass','mwrc5fusedILi32ELi128ELi4ELi1ELb1ELi1ELb1')]:
  parts=[p for p in path.read_text().split('Function : ')[1:] if marker in p.splitlines()[0]];assert len(parts)==1,(label,len(parts));ins=parse(parts[0]);loops=[]
  for pc,op,args,pred in ins:
   m=re.search(r'0x([0-9a-f]+)',args)
   if op=='BRA' and m and int(m[1],16)<pc:
    lo=int(m[1],16);body=[i for i in ins if lo<=i[0]<=pc]
    if sum(i[1].startswith('HGMMA.') for i in body)==8:loops.append((pc-lo,lo,pc,body))
  _,lo,hi,body=min(loops);ops=collections.Counter(op for _,op,_,_ in body);cat=collections.Counter()
  for _,op,arg,pred in body:
   if op.startswith(('HGMMA','WARPGROUP','FENCE.VIEW')):kind='mma_fence_commit_wait'
   elif op.startswith(('HMUL','LOP3','SHF','PRMT','FLO')):kind='unpack_scale_bitops'
   elif op.startswith(('LDS','STS','LDG','STG')):kind='memory_instructions'
   elif op.startswith(('IADD','IMAD','LEA','UIADD','VIADD','ULEA','USHF','ULOP','UISETP')):kind='address_index_integer'
   else:kind='control_sync_other'
   cat[kind]+=1
  row=dict(label=label,loop=[hex(lo),hex(hi)],static_instructions=len(body),per16_weights={k:v/4 for k,v in cat.items()},instructions_per_weight=len(body)/64,uniform_instructions=sum(op.startswith(('U','VIADD')) for _,op,_,_ in body),opcodes=dict(ops),conditional_sites=[(hex(pc),op,arg,pred) for pc,op,arg,pred in body if pred],note='Static loop includes both cp.async drain paths and warp-masked SF copies; executed count depends on branch. Categories use opcodes, not a data-flow attribution. K128 is64 weights/lane, four16-weight groups.')
  rows.append(row);(H/(label+'_loop.txt')).write_text('\n'.join(f'{pc:04x} {pred:5s} {op:36s} {arg}' for pc,op,arg,pred in body)+'\n')
 (H/'baseline_counts.json').write_text(json.dumps(rows,indent=2)+'\n')
 for r in rows:print(r['label'],r['loop'],r['static_instructions'],r['instructions_per_weight'],r['per16_weights'],r['opcodes'])

if __name__=='__main__':main()
