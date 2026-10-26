"""Count-only coarse templates and owner-sharded DSM budgets; no runnable DSM arm."""
import ast, hashlib, json, re, subprocess
from pathlib import Path
D = Path(__file__).resolve().parent
R = D.parents[5]
O = R/'runs/screen/p3a_parking'
O.mkdir(parents=True,exist_ok=True)
src = next(ast.literal_eval(n.value) for n in ast.parse((D/'slice_cell.py').read_text()).body
           if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='DEP_CUDA')
for token in ('GO','INFO','ATTR'):
    anchor=f' {token}(128,2)'
    assert src.count(anchor)==1
    src=src.replace(anchor,anchor+f' {token}(256,2) {token}(512,2)')
p=O/'coarse_count.cu'
p.write_text(src)
cmd=['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-c',str(p),'-o',str(p.with_suffix('.o')),'-Xptxas=-v']
z=subprocess.run(cmd,text=True,capture_output=True)
log=z.stdout+z.stderr
p.with_suffix('.log').write_text(log)
z.check_returncode()
cpp=O/'occupancy.cpp'
cpp.write_text('''#include <cuda_occupancy.h>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){cudaOccDeviceProp p;p.computeMajor=9;p.computeMinor=0;p.maxThreadsPerBlock=1024;p.maxThreadsPerMultiprocessor=2048;p.regsPerBlock=65536;p.regsPerMultiprocessor=65536;p.warpSize=32;p.sharedMemPerBlock=49152;p.sharedMemPerMultiprocessor=233472;p.numSms=132;p.sharedMemPerBlockOptin=232448;p.reservedSharedMemPerBlock=1024;cudaOccFuncAttributes a;a.maxThreadsPerBlock=1024;a.numRegs=atoi(argv[1]);a.sharedSizeBytes=0;a.shmemLimitConfig=FUNC_SHMEM_LIMIT_OPTIN;a.maxDynamicSharedSizeBytes=232448;cudaOccDeviceState s;cudaOccResult r{};auto e=cudaOccMaxActiveBlocksPerMultiprocessor(&r,&p,&a,&s,256,atoi(argv[2]));if(e)return e;printf("%d %zu %d\\n",r.activeBlocksPerMultiprocessor,r.allocatedSharedMemPerBlock,r.allocatedRegistersPerBlock);}
''')
subprocess.run(['g++','-O2','-std=c++17','-I/usr/local/cuda/include',str(cpp),'-o',str(cpp.with_suffix(''))],check=True,capture_output=True)
def occ(regs,smem):
    v=list(map(int,subprocess.check_output([str(cpp.with_suffix('')),str(regs),str(smem)],text=True).split()))
    return dict(CTAs_per_SM=v[0],allocated_shared_bytes=v[1],allocated_registers=v[2])
rows=[]
for S,C in ((256,16),(512,8)):
    m=re.search(r'Function properties for \S*slice_mainILi'+str(S)+r'ELi2E\S*\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers',log)
    assert m
    stack,stores,loads,regs=map(int,m.groups())
    assert regs==108 and stack==stores==loads==0
    basic=84624+16*S
    # Aggregate owner output is8tokens*4096cols*4B,partitioned over C ranks.
    dsm=basic+8*4096*4//C+256
    rows.append(dict(slice=S,cluster_blocks=C,math_count_only=dict(registers=regs,stack=stack,spill_stores=stores,spill_loads=loads,shared_bytes=basic,**occ(regs,basic)),
                     owner_sharded_DSM_model=dict(shared_bytes=dsm,extra_register_headroom_to128=128-regs,**occ(regs,dsm)),
                     complete_local_partial_minimum=dict(shared_bytes=2*36864+8*4096*4,**occ(regs,2*36864+8*4096*4)),
                     cluster_occupancy='Unmeasured;CTA calculator is not cudaOccupancyMaxActiveClusters',
                     implementation='Coarse NORED math instantiated only;no DSM correctness/performance implementation'))
out=dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),count_command=cmd,ptxas_log_sha256=hashlib.sha256(log.encode()).hexdigest(),calculator='CUDA13 cuda_occupancy.h,sm90 132SM/65536regs/233472B shared/1024B reserved per CTA,256threads',rows=rows)
(D/'sizing_counts.json').write_text(json.dumps(out,indent=2)+'\n')
for v in rows:print(v)
