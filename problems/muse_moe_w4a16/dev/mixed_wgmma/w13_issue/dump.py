"""Compile and count the W13-only variants for SM90a before running a lab."""
import subprocess,re,json,collections
from pathlib import Path
from count import parse
H=Path(__file__).resolve().parent

def main():
 s=(H/'issue.cu').read_text();s=s[:s.index('int64_t wi_init')]
 for r,v in ((False,False),(False,True),(True,True)):
  s+=f'template __global__ void mwwi::fused<32,128,4,1,true,1,{str(r).lower()},{str(v).lower()}>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const int*,const __nv_bfloat16*,float*,__nv_bfloat16*,float*,float*,float*,int*,__nv_bfloat16*,int,int,int,bool);\n'
 for nt,filt in ((16,1),(32,2)):
  s+=f'template __global__ void mwwi::fused<{nt},128,4,1,true,1,true,true,{filt}>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const int*,const __nv_bfloat16*,float*,__nv_bfloat16*,float*,float*,float*,int*,__nv_bfloat16*,int,int,int,bool);\n'
 for nt,r,filt in ((32,False,0),(32,True,0),(16,True,1),(32,True,2)):
  s+=f'template __global__ void mwwi::fused<{nt},128,4,1,true,1,{str(r).lower()},true,{filt},true>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const int*,const __nv_bfloat16*,float*,__nv_bfloat16*,float*,float*,float*,int*,__nv_bfloat16*,int,int,int,bool);\n'
 path=H/'native.cu';path.write_text(s);p=subprocess.run(['nvcc','--cubin','-O3','-lineinfo','-arch=sm_90a','-std=c++20','-Xptxas=-v',str(path),'-o',str(path.with_suffix('.cubin'))],capture_output=True,text=True);(H/'native.ptxas.log').write_text(p.stdout+p.stderr);p.check_returncode();assert 'C7513' not in p.stderr
 sass=subprocess.run(['cuobjdump','--dump-sass',str(path.with_suffix('.cubin'))],capture_output=True,text=True,check=True).stdout;(H/'native.sass').write_text(sass);out=[]
 for part in sass.split('Function : ')[1:]:
  ins=parse(part);loops=[]
  for pc,op,args,pred in ins:
   m=re.search(r'0x([0-9a-f]+)',args)
   if op=='BRA' and m and int(m[1],16)<pc:
    lo=int(m[1],16);body=[i for i in ins if lo<=i[0]<=pc]
    if sum(i[1].startswith('HGMMA.') for i in body)==8:loops.append((pc-lo,lo,pc,body))
  _,lo,hi,body=min(loops);ops=collections.Counter(i[1] for i in body);row=dict(name=part.splitlines()[0].strip(),loop=[hex(lo),hex(hi)],static_instructions=len(body),per_weight=len(body)/64,opcodes=dict(ops),local_memory=sum(i[1].startswith(('LDL','STL')) for i in ins),waits=[i[2] for i in ins if i[1].startswith('WARPGROUP.DEPBAR')]);assert row['local_memory']==0;assert len(row['waits'])==3 and row['waits'][-1].endswith('0x0'),row;out.append(row)
 (H/'counts.json').write_text(json.dumps(out,indent=2)+'\n')
 for row in out:print(row)

if __name__=='__main__':main()
