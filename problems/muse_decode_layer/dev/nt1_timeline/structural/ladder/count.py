import collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_ladder';cells=json.loads((D/'cells.json').read_text());rows=[]
cpp=O/'occ.cpp';cpp.write_text('''#include <cuda_occupancy.h>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){cudaOccDeviceProp p;p.computeMajor=9;p.computeMinor=0;p.maxThreadsPerBlock=1024;p.maxThreadsPerMultiprocessor=2048;p.regsPerBlock=65536;p.regsPerMultiprocessor=65536;p.warpSize=32;p.sharedMemPerBlock=49152;p.sharedMemPerMultiprocessor=233472;p.numSms=132;p.sharedMemPerBlockOptin=232448;p.reservedSharedMemPerBlock=1024;cudaOccFuncAttributes a;a.maxThreadsPerBlock=1024;a.numRegs=atoi(argv[1]);a.sharedSizeBytes=atoi(argv[3]);a.shmemLimitConfig=FUNC_SHMEM_LIMIT_OPTIN;a.maxDynamicSharedSizeBytes=232448-a.sharedSizeBytes;cudaOccDeviceState s;cudaOccResult r{};auto e=cudaOccMaxActiveBlocksPerMultiprocessor(&r,&p,&a,&s,256,atoi(argv[2]));if(e)return e;printf("%d\\n",r.activeBlocksPerMultiprocessor);}
''');subprocess.run(['g++','-O2','-I/usr/local/cuda/include',str(cpp),'-o',str(cpp.with_suffix(''))],check=True)
for i in range(3):
 p=O/f'lad{i}.cu';assert hashlib.sha256(p.read_bytes()).hexdigest()==cells[str(i)]['cuda_sha256'];z=subprocess.run(['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-c',str(p),'-o',str(p.with_suffix('.o')),'-Xptxas=-v'],text=True,capture_output=True);log=z.stdout+z.stderr;p.with_suffix('.compile.log').write_text(log);z.check_returncode();sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
 for fn,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
  f=95 if i==0 else 69;m=re.search(r'lad_varILi1ELi'+str(f)+r'ELi(\d+)ELb(\d)E',fn);base=re.search(r'moe_varILi1ELi'+str(f)+r'E',fn)
  if not m and not base:continue
  arm='BASE' if base else 'HIST' if int(m[2]) else {0:'L0S',10:'L0',11:'LOOP',1:'L1',2:'L2',3:'L3',5:'MATH',6:'META',7:'ACT',8:'STREAM',9:'BSINK'}[int(m[1])];v=re.search(r'Function properties for '+re.escape(fn)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers([^\n]*)',log);assert v
  stack,stores,loads,regs=map(int,v.groups()[:4]);sm=re.search(r'(\d+) bytes smem',v[5]);stat=int(sm[1]) if sm else 0;ops=dict(collections.Counter(x.split('.')[0] for x in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',body)))
  for M in ([1] if i==0 else [8] if i==1 else [16,32]):
   dyn=2*36864+8192+32+1904+((M*8+15)&~15)+4*M*8;resident=int(subprocess.check_output([str(cpp.with_suffix('')),str(regs),str(dyn),str(stat)],text=True));assert resident==2,(M,arm,regs,stat);assert stores==loads==0,(M,arm,stores,loads)
   row=dict(M=M,arm=arm,registers=regs,stack=stack,spill_stores=stores,spill_loads=loads,shared=dyn,static=stat,resident=resident,ops=ops,cell_sha256=cells[str(i)]['cell_sha256'],cuda_sha256=cells[str(i)]['cuda_sha256']);rows.append(row)
  print(i,arm,regs,'HMUL/HFMA/HMMA',*[ops.get(x,0) for x in ['HMUL2','HFMA2','HMMA']],flush=True)
for M in (1,8,16,32):
 rr={v['arm']:v for v in rows if v['M']==M};assert len(rr)==13
 for arm in ('L2','L3','BASE','BSINK','MATH','HIST'):assert rr[arm]['ops']['HMMA']==32 and rr[arm]['ops'].get('HMUL2',0)+rr[arm]['ops'].get('HFMA2',0)>=128
 assert rr['L1']['ops'].get('HMMA',0)==rr['L0']['ops'].get('HMMA',0)==0
 assert rr['L1']['ops'].get('HMUL2',0)+rr['L1']['ops'].get('HFMA2',0)>=128
 assert rr['L0']['ops'].get('HMUL2',0)==0
 assert rr['MATH']['ops'].get('UBLKCP',0)==0
 assert rr['L3']['ops'].get('NANOSLEEP',0)==0
(D/'counts.json').write_text(json.dumps(dict(rows=rows,native_codegen='sm90a',all_twoCTA=True,all_zero_spills=True,sink_note='L1 may use64HMUL2+64equivalentBF16 HFMA2 rather than128HMUL2;all128scaled values kept byvolatileSTS;BSINK calibrates'),indent=2)+'\n')
