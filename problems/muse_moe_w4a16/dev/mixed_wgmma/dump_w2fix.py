"""Native SM90a audit of each W2 repair and its mainloop waits."""
import hashlib,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent

def main():
 full=(H/'w2fix.cu').read_text();src='#include <cuda_bf16.h>\n#include <stdint.h>\n'+full[full.index('namespace mwfx'):full.index('#define FX_CASES')]+'\n}\n'
 for nt in (16,32):
  for first,mode in (((True,1),(False,0),(False,1)) if nt==16 else ((True,1),(False,1))):
   src+=f'template __global__ void mwfx::fused<{nt},128,4,1,{str(first).lower()},{mode}>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const int*,const __nv_bfloat16*,float*,__nv_bfloat16*,float*,float*,float*,int*,__nv_bfloat16*,int,int,int,bool);\n'
 a=full.index('namespace mwfx {\n__global__ void finish');b=full.index('void fx_finish',a);src+=full[a:b]
 path=H/'diagnostics/w2fix.cu';path.write_text(src);p=subprocess.run(['nvcc','--cubin','-O3','-lineinfo','-arch=sm_90a','-std=c++20','-Xptxas=-v',str(path),'-o',str(path.with_suffix('.cubin'))],capture_output=True,text=True);(H/'diagnostics/w2fix.ptxas.log').write_text(p.stdout+p.stderr);p.check_returncode();assert 'C7513' not in p.stderr
 sass=subprocess.run(['cuobjdump','--dump-sass',str(path.with_suffix('.cubin'))],capture_output=True,text=True,check=True).stdout;path.with_suffix('.sass').write_text(sass);regs={};name=None
 for line in p.stderr.splitlines():
  if 'Function properties for ' in line:name=line.split('Function properties for ',1)[1].strip()
  m=re.search(r'Used (\d+) registers',line)
  if m:regs[name]=int(m[1])
 rows=[]
 for part in sass.split('Function : ')[1:]:
  name=part.splitlines()[0].strip();ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;',part);waits=[arg for op,arg in ops if op.startswith('WARPGROUP.DEPBAR')]
  row=dict(name=name,registers=regs[name],local_memory=sum(op.startswith(('LDL','STL')) for op,_ in ops),mma=sum(op.startswith('HGMMA.') for op,_ in ops),waits=waits,global_atomic=sum(op.startswith(('ATOM','REDG')) for op,_ in ops))
  assert row['local_memory']==0,row
  if 'fused' in name:assert len(waits)==3 and waits[0].endswith('0x1') and waits[1].endswith('0x1') and waits[2].endswith('0x0'),row
  rows.append(row)
 assert len(rows)==7 and sum('fused' in r['name'] for r in rows)==5 and sum('finish' in r['name'] for r in rows)==1
 report=dict(sha256=hashlib.sha256(full.encode()).hexdigest(),kernels=rows);(H/'diagnostics/w2fix_sass_counts.json').write_text(json.dumps(report,indent=2)+'\n');print('W2FIX_NATIVE_PASS',[(r['registers'],r['global_atomic']) for r in rows])

if __name__=='__main__':main()
