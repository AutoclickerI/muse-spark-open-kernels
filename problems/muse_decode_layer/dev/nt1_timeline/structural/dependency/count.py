"""Exact sm90a SASS/resources and offline H200 occupancy,before any H200 run."""
import ast,collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_dependency';cells=json.loads((D/'cells.json').read_text());result={}
header=(O/'slice.cu').read_text();route=header[header.index('struct PrefetchState'):header.index('// unit ->')];layout='template<int S>constexpr int slice_smem(){return 2*TILE_BYTES+8192+32+sizeof(Route)+256+512+8*S*2;}'
deep_layout=re.search(r'template<int S>constexpr int slice_smem\(\)\{.*?\}',(O/'deep.cu').read_text())[0].replace('slice_smem','deep_smem')
cpp='''#include <cuda_occupancy.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
constexpr int NCW=8,NLOC=32,TILE_BYTES=36864;
'''+route+layout+deep_layout+'''
int main(int argc,char**argv){int regs=atoi(argv[1]),stat=atoi(argv[2]),dyn=atoi(argv[3]);cudaOccDeviceProp p;p.computeMajor=9;p.computeMinor=0;p.maxThreadsPerBlock=1024;p.maxThreadsPerMultiprocessor=2048;p.regsPerBlock=65536;p.regsPerMultiprocessor=65536;p.warpSize=32;p.sharedMemPerBlock=49152;p.sharedMemPerMultiprocessor=233472;p.numSms=132;p.sharedMemPerBlockOptin=232448;p.reservedSharedMemPerBlock=1024;cudaOccFuncAttributes a;a.maxThreadsPerBlock=1024;a.numRegs=regs;a.sharedSizeBytes=stat;a.shmemLimitConfig=FUNC_SHMEM_LIMIT_OPTIN;a.maxDynamicSharedSizeBytes=232448-stat;cudaOccDeviceState st;cudaOccResult r{};if(dyn<0)dyn=dyn==-64?slice_smem<64>():dyn==-1128?deep_smem<128>():slice_smem<128>();auto e=cudaOccMaxActiveBlocksPerMultiprocessor(&r,&p,&a,&st,256,dyn);if(e)return e;printf("%zu %d %d %d %zu %d\\n",sizeof(Route),dyn,regs,r.activeBlocksPerMultiprocessor,r.allocatedSharedMemPerBlock,r.allocatedRegistersPerBlock);}
'''
p=O/'budget.cpp';p.write_text(cpp);subprocess.run(['g++','-O2','-std=c++17','-I/usr/local/cuda/include',str(p),'-o',str(p.with_suffix(''))],check=True,capture_output=True)
for name,spec in cells.items():
 if name in ('SLICE','DEEP'):cu=O/('slice.cu' if name=='SLICE' else 'deep.cu')
 else:
  text=(R/spec['cell']).read_text();src=next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CUDA'));cu=O/(name.lower()+'.cu');cu.write_text(src)
 assert hashlib.sha256(cu.read_bytes()).hexdigest()==spec['cuda_sha256']
 if name not in ('SLICE','DEEP'):
  z=subprocess.run(['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-c',str(cu),'-o',str(cu.with_suffix('.o')),'-Xptxas=-v'],capture_output=True,text=True);cu.with_suffix('.compile.log').write_text(z.stdout+z.stderr);z.check_returncode()
 log=cu.with_suffix('.compile.log').read_text();sass=subprocess.check_output(['cuobjdump','--dump-sass',str(cu.with_suffix('.o'))],text=True);cu.with_suffix('.sass').write_text(sass);parts={}
 for fn,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
  matched=re.search(r'slice_mainILi(64|128)ELi(0|1|2)E',fn) if name in ('SLICE','DEEP') else re.search(r'moe_varILi1ELi69E',fn)
  if not matched:continue
  key=str(matched[1])+'_'+matched[2] if name in ('SLICE','DEEP') else '69';res=re.search(r'Function properties for '+re.escape(fn)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers([^\n]*)',log);assert res
  stack,stores,loads,regs=map(int,res.groups()[:4]);sm=re.search(r'(\d+) bytes smem',res[5]);static=int(sm[1]) if sm else 0;ops=dict(collections.Counter(m[1].split('.')[0] for m in re.finditer(r'/\*[0-9a-f]+\*/\s+(?:@!?(?:U?P)\d+\s+)?([A-Z][A-Z0-9_.]*)',body)))
  rows=[]
  for M in (8,16,32):
   dynamic=(-1128 if name=='DEEP' else -int(matched[1])) if name in ('SLICE','DEEP') else 2*36864+8192+32+1904+((M*8+15)&~15)+4*M*8
   vals=list(map(int,subprocess.check_output([str(p.with_suffix('')),str(regs),str(static),str(dynamic)],text=True).split()));routeb,dyn,reg,ctas,allocsm,allocreg=vals;assert ctas==2,(name,key,vals);assert stores==loads==0,(name,key,stores,loads);rows.append(dict(M=M,registers=regs,stack=stack,spill_stores=stores,spill_loads=loads,dynamic=dyn,static=static,resident=ctas,allocated_smem=allocsm,allocated_registers=allocreg))
  parts[key]=dict(function=fn,ops=ops,rows=rows)
  print(name,key,regs,stack,stores,loads,rows[0]['dynamic'],'2CTA',flush=True)
 result[name]=dict(cell_sha256=spec['cell_sha256'],cuda_sha256=spec['cuda_sha256'],parts=parts)
 (D/'counts.json').write_text(json.dumps(dict(calculator='CUDA13 cuda_occupancy.h,H200 sm90,233472shared/65536regs perSM,1024Breserved/block,256threads',cells=result),indent=2)+'\n')
# NOWAIT's matched decode/MMA region is byte-identical;only two spin loops differ.
for i in (1,2):
 def core(cu):a=cu.index('    const uint8_t* tile =');b=cu.index('    __syncwarp();',a);return cu[a:b]
 assert core((O/f'base{i}.cu').read_text())==core((O/f'nowait{i}.cu').read_text())
for mode in ('128_0','128_1','128_2'):
 ops=result['SLICE']['parts'][mode]['ops'];assert ops.get('HMMA',0)>0 and not ops.get('NANOSLEEP',0)
print('MATCHED_ORACLE_AND_NO_GLOBAL_SPIN_PASS',flush=True)
