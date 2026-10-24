"""Audit actual SM90a raw/recoded specializations and compare scale-conversion opcodes."""
import hashlib,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent

def main():
 full=(H/'recode.cu').read_text();src='#include <cuda_bf16.h>\n#include <stdint.h>\n'+full[full.index('namespace mwrc'):full.index('#define RC_CASES')]+'\n}\n'
 for nt in (16,32):
  for first in (True,False):
   for rec in (True,False):src+=f'template __global__ void mwrc::fused<{nt},128,4,1,{str(first).lower()},1,{str(rec).lower()}>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const int*,const __nv_bfloat16*,float*,__nv_bfloat16*,float*,float*,float*,int*,__nv_bfloat16*,int,int,int,bool);\n'
 a=full.index('namespace mwrc {\n__global__ void finish');b=full.index('void rc_finish',a);src+=full[a:b]
 native=(H/'nt1.cu').read_text();a=native.index('namespace n1');b=native.index('int64_t n1_init',a);src+=native[a:b]+'\ntemplate __global__ void n1::moe_fused<1>(const __nv_bfloat16*,const int*,const float*,const __nv_bfloat16*,const uint8_t*,const uint8_t*,const float*,const float*,__nv_bfloat16*,float*,float*,int*,__nv_bfloat16*,int,int,int);\n'
 path=H/'native.cu';path.write_text(src);p=subprocess.run(['nvcc','--cubin','-O3','-lineinfo','-arch=sm_90a','-std=c++20','-Xptxas=-v',str(path),'-o',str(path.with_suffix('.cubin'))],capture_output=True,text=True);(H/'native.ptxas.log').write_text(p.stdout+p.stderr);p.check_returncode();assert 'C7513' not in p.stderr
 sass=subprocess.run(['cuobjdump','--dump-sass',str(path.with_suffix('.cubin'))],capture_output=True,text=True,check=True).stdout;(H/'native.sass').write_text(sass);regs={};name=None
 for line in p.stderr.splitlines():
  if 'Function properties for ' in line:name=line.split('Function properties for ',1)[1].strip()
  m=re.search(r'Used (\d+) registers',line)
  if m:regs[name]=int(m[1])
 rows=[]
 for part in sass.split('Function : ')[1:]:
  name=part.splitlines()[0].strip();ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;',part);waits=[a for op,a in ops if op.startswith('WARPGROUP.DEPBAR')]
  r=dict(name=name,registers=regs[name],local_memory=sum(op.startswith(('LDL','STL')) for op,_ in ops),mma=sum(op.startswith('HGMMA.') for op,_ in ops),waits=waits,shared_load=sum(op.startswith('LDS') for op,_ in ops),prmt=sum(op.startswith('PRMT') for op,_ in ops))
  if name.startswith('_ZN4mwrc5fused'):
   assert r['local_memory']==0,r
   assert len(waits)==3 and waits[0].endswith('0x1') and waits[1].endswith('0x1') and waits[2].endswith('0x0'),r
  rows.append(r)
 assert sum('mwrc5fused' in r['name'] for r in rows)==8
 (H/'sass_counts.json').write_text(json.dumps(dict(cuda_sha256=hashlib.sha256(full.encode()).hexdigest(),nt1_sha256=hashlib.sha256(native.encode()).hexdigest(),kernels=rows),indent=2)+'\n');print('NATIVE_RECODE_PASS',[(r['registers'],r['local_memory'],r['shared_load'],r['prmt']) for r in rows])

if __name__=='__main__':main()
