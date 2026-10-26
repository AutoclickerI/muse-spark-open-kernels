"""Cheapest pre-run screen:exact sm90a ptxas resources and CUDA's offline H200 occupancy calculator."""
import ast,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_overlap';cells=json.loads((D/'cells.json').read_text());out=json.loads((D/'counts.json').read_text())['cells'] if (D/'counts.json').exists() else {}

for name,spec in cells.items():
 if name in out and out[name]['cuda_sha256']==spec['cuda_sha256']:continue
 if name.startswith('BASE'):
  cell=(R/spec['cell']).read_text();cu=next(ast.literal_eval(n.value) for n in ast.parse(cell).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CUDA'));(O/(name.lower()+'.cu')).write_text(cu)
 cu=O/(name.lower()+'.cu');obj=cu.with_suffix('.o');text=cu.read_text();assert hashlib.sha256(text.encode()).hexdigest()==spec['cuda_sha256']
 p=subprocess.run(['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-c',str(cu),'-o',str(obj),'-Xptxas=-v'],capture_output=True,text=True);cu.with_suffix('.compile.log').write_text(p.stdout+p.stderr);p.check_returncode()
 rr=re.search(r'Function properties for (\S*moe_varILi1ELi69E\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers([^\n]*)',p.stderr);assert rr,p.stderr
 fn,stack,stores,loads,regs,rest=rr.groups();stack,stores,loads,regs=map(int,(stack,stores,loads,regs));sm=re.search(r'(\d+) bytes smem',rest);static=int(sm[1]) if sm else 0
 route=text[text.index('struct PrefetchState'):text.index('// unit ->')];a=text.index('size_t smem_bytes');b=text.index('\n}',a)+2;layout=text[a:b]
 cpp='''#include <cuda_occupancy.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
constexpr int NCW=NCW_VALUE,NLOC=32,TILE_BYTES=36864;
'''+route+layout+'''
int main(int argc,char**argv){int regs=atoi(argv[1]),stat=atoi(argv[2]),threads=atoi(argv[3]);cudaOccDeviceProp p;p.computeMajor=9;p.computeMinor=0;p.maxThreadsPerBlock=1024;p.maxThreadsPerMultiprocessor=2048;p.regsPerBlock=65536;p.regsPerMultiprocessor=65536;p.warpSize=32;p.sharedMemPerBlock=49152;p.sharedMemPerMultiprocessor=233472;p.numSms=132;p.sharedMemPerBlockOptin=232448;p.reservedSharedMemPerBlock=1024;
 cudaOccFuncAttributes a;a.maxThreadsPerBlock=1024;a.numRegs=regs;a.sharedSizeBytes=stat;a.shmemLimitConfig=FUNC_SHMEM_LIMIT_OPTIN;a.maxDynamicSharedSizeBytes=232448-stat;cudaOccDeviceState st;
 for(int m:{8,16,32})for(int depth:{1,2}){cudaOccResult r{};size_t n=smem_bytes(1,depth,m);auto e=cudaOccMaxActiveBlocksPerMultiprocessor(&r,&p,&a,&st,threads,n);if(e)return e;printf("%d %d %zu %zu %d %d %d %d %d %zu\\n",m,depth,sizeof(Route),n,regs,r.activeBlocksPerMultiprocessor,r.blockLimitRegs,r.blockLimitSharedMem,r.allocatedRegistersPerBlock,r.allocatedSharedMemPerBlock);}}
'''
 cpp=cpp.replace('NCW_VALUE',re.search(r'constexpr int NCW = (\d+);',text)[1]);cpp=cpp.replace('#include <cstdlib>','#include <cstdlib>\n#include <initializer_list>');cp=O/(name.lower()+'_budget.cpp');cp.write_text(cpp);exe=cp.with_suffix('');subprocess.run(['g++','-O2','-std=c++17','-I/usr/local/cuda/include',str(cp),'-o',str(exe)],check=True,capture_output=True);raw=subprocess.check_output([str(exe),str(regs),str(static),str(spec['threads'])],text=True);rows=[]
 for line in raw.splitlines():
  m,depth,route_bytes,dynamic,nreg,ctas,reglimit,smlimit,allocreg,allocsm=map(int,line.split());rows.append(dict(M=m,depth=depth,route_bytes=route_bytes,dynamic_smem=dynamic,static_smem=static,registers=nreg,resident=ctas,register_limit=reglimit,smem_limit=smlimit,allocated_registers=allocreg,allocated_smem_including_driver=allocsm))
 out[name]=dict(cuda_sha256=spec['cuda_sha256'],cell_sha256=spec['cell_sha256'],registers=regs,stack_bytes=stack,spill_stores=stores,spill_loads=loads,static_smem=static,threads=spec['threads'],rows=rows)
 (D/'counts.json').write_text(json.dumps(dict(calculator='CUDA13 cuda_occupancy.h offline;H200 sm90 132SM,65536regs/SM,233472smem/SM,1024reserved/block,role-specific256/384threads',header_sha256=hashlib.sha256(Path('/usr/local/cuda/include/cuda_occupancy.h').read_bytes()).hexdigest(),cells=out),indent=2)+'\n');print(name,'regs',regs,'stack/spills',stack,stores,loads,'depth2',[(v['M'],v['dynamic_smem'],v['resident']) for v in rows if v['depth']==2],flush=True)
