"""Reject writes to live register-A operands across the actual repeated K-loop SASS."""
import re,json
from pathlib import Path
from count import parse
H=Path(__file__).resolve().parent

def main():
 rows=[]
 for part in (H/'native.sass').read_text().split('Function : ')[1:]:
  name=part.splitlines()[0].strip();ins=parse(part);loops=[]
  for pc,op,arg,pred in ins:
   m=re.search(r'0x([0-9a-f]+)',arg)
   if op=='BRA' and m and int(m[1],16)<pc:
    lo=int(m[1],16);body=[i for i in ins if lo<=i[0]<=pc]
    if sum(i[1].startswith('HGMMA.') for i in body)==8:loops.append((pc-lo,body))
  _,body=min(loops);pending=[];group=set();hazards=[];groups=[]
  for k in range(32):
   for pc,op,arg,pred in body:
    if op.startswith('HGMMA.'):
     m=re.match(r'R\d+, R(\d+),',arg);assert m,(op,arg);r=int(m[1]);group.update(range(r,r+4))
     if 'gsb0' in arg:
      pending.append(group);groups.append(sorted(group));group=set()
    elif op.startswith('WARPGROUP.DEPBAR'):
     n=int(re.search(r'0x([0-9a-f]+)$',arg)[1],16);pending=pending[-n:] if n else []
    elif pred!='@!PT':
     m=re.match(r'R(\d+)',arg)
     if m:
      r=int(m[1]);width=4 if '.128' in op else 2 if '.64' in op else 1;dest=set(range(r,r+width));live=set().union(group,*pending)
      if dest&live:hazards.append((k,hex(pc),op,arg,sorted(dest&live)))
  assert not hazards,(name,hazards[:3])
  rows.append(dict(name=name,group0=groups[0],group1=groups[1],simulated_groups=len(groups),early_overwrites=0))
 assert len(rows)==9
 n=0
 for nt,static in ((16,2048),(32,4096),(32,5120)):
  base=static+18432
  assert base%1024==0
  for stage in range(4):
   for chunk in range(2):
    b=base+(stage*nt*128+chunk*nt*64)*2
    for row in range(nt):
     for k in range(64):
      logical=b+row*128+k*2;reader=logical^(((logical>>7)&7)<<4);writer=b+(row*64+(k^((row&7)*8)))*2
      assert reader==writer and b<=reader<b+nt*128;n+=1
 (H/'lifetime.json').write_text(json.dumps(dict(kernels=rows,B128_addresses=n),indent=2)+'\n');print('LIFETIME_PASS',len(rows),n)

if __name__=='__main__':main()
