"""Reproduce the pre-H200 two-base decoder rejection and exhaustive finite-format identity."""
import collections,hashlib,json,math,re,struct,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[4];source=D/'two_base_probe.cu';obj=R/'runs/screen/p3a_two_base.o'
r=subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_90a','-c',str(source),'-o',str(obj),'-Xptxas=-v'],capture_output=True,text=True,check=True)
sass=subprocess.check_output(['cuobjdump','--dump-sass',str(obj)],text=True);rows={};ints={'LOP3','SHF','SHL','SHR','IADD3','IADD','IMAD','ISETP','SEL','PRMT','BFE','BFI','LEA','PLOP3','UIADD3','ULOP3','USHF','UISETP','UIMAD','VIADD'}
for name,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
 ops=collections.Counter(k.split('.')[0] for k in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]*)',body));tag='two_base' if '15two_base_decodeE' in name else 'old'
 regs=int(re.search(r'Function properties for '+re.escape(name)+r'.*?Used (\d+) registers',r.stderr,re.S)[1])
 rows[tag]=dict(integer_instructions=sum(v for k,v in ops.items() if k in ints),registers=regs,ops=dict(ops),function=name)
def bits(x):
 u=struct.unpack('<I',struct.pack('<f',x))[0];return (u+0x7fff+((u>>16)&1))>>16
count=0
for raw in range(127):
 exp,m=raw>>3,raw&7;s=m*2**-9 if exp==0 else (1+m/8)*2**(exp-7)
 for scale_sign in (-1,1):
  for code in range(16):
   mag=code&7;sgn=(-1 if code&8 else 1)*scale_sign
   value=(0.,.5,1.,1.5,2.,3.,4.,6.)[mag]*s/256
   if mag and s:
    base=bits((1.5 if mag&1 and mag!=1 else 1.)*s/256)
    assembled=base+((mag>>1)-1)*128
    assert assembled==bits(value),(raw,code,assembled,bits(value))
    assert (assembled|(0x8000 if sgn<0 else 0))==bits(sgn*value)
   else:assert value==0
   # Both positions of a BF16x2 register use exactly this identity.
   count+=2
assert rows['two_base']['integer_instructions']>rows['old']['integer_instructions']
report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),architecture='sm90a',same_work='16 output BF16x2 registers per lane, original recoded code words and scale input',finite_format_packed_half_checks=count,rows=rows,decision='Dropped before any H200 run: integer issue grows. This closes this scalar two-base assembly implementation, not every possible packed exponent decoder.')
(D/'decode_issue.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
