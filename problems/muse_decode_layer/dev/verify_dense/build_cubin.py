from pathlib import Path
import subprocess,json,hashlib,re,zlib,base64
HERE=Path(__file__).resolve().parent
source=(HERE/'cutlass.cu').read_text();tag=hashlib.sha256(source.encode()).hexdigest()[:14]
root=Path.home()/'.cache/torch_extensions/py312_cu130';lib=root/('muse_vdense_cutlass_'+tag)/('muse_vdense_cutlass_'+tag+'.so')
if not lib.exists():raise RuntimeError('compile_cutlass.py first')
subprocess.run(['/usr/local/cuda/bin/cuobjdump','--extract-elf','all',str(lib)],cwd=HERE,check=True,capture_output=True)
cubin=next(HERE.glob('muse_vdense_cutlass_*.cubin'));data=cubin.read_bytes()
sass=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(cubin)],capture_output=True,text=True,check=True).stdout
(HERE/'cutlass.sass.log').write_text(sass)
names=re.findall(r'Function : (\S+)',sass)
assert len(names)==6,names
# Match config from template tuple:128x32x128 cluster1,1 StreamK;
#64x64x128 cluster1,2 persistent;128x64x64 cluster2,1 StreamK.
functions={}
for name in names:
 config=0 if 'MMA_64x32' in name else 1 if 'Pingpong' in name else 2
 fp32='LinearCombinationIffff' in name
 key=f'{config}:{int(fp32)}';assert key not in functions,key;functions[key]=name
encoded=base64.b64encode(zlib.compress(data,9)).decode()
(HERE/'cubin.py.inc').write_text('VD_CUBIN='+repr(encoded)+'\nVD_CUBIN_FUNCTIONS='+repr(functions)+'\n')
info=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),cubin_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),compressed_bytes=len(encoded),functions=functions,sass_counts={n:len(re.findall(n,sass)) for n in ('HGMMA','UTMALDG','UTMASTG','WARPGROUP','SYNCS')},native_execution=False)
(HERE/'cubin.json').write_text(json.dumps(info,indent=2)+'\n');print({k:v for k,v in info.items() if k!='functions'})
# CUTLASS's original host argument conversion/workspace init, compiled as C++.
host=source.replace('#include <cuda_runtime.h>','#include <cuda_runtime.h>\n#include <cuda.h>\n#include <cstring>',1)
a=host.index('    auto a=args(w,x,y,m,n,k,mode);Device op;');b=host.index('\n  }\n};',a)
host=host[:a]+'''    auto a=args(w,x,y,m,n,k,mode);
    if(!Kernel::can_implement(a))return 100;
    *needed=Kernel::get_workspace_size(a);if(!launch)return 0;if(*needed>capacity)return -2;
    auto status=Kernel::initialize_workspace(a,workspace,stream);
    if(status!=cutlass::Status::kSuccess)return 200+(int)status;
    auto params=Kernel::to_underlying_arguments(a,workspace);
    auto grid=Kernel::get_grid_shape(params);auto block=Kernel::get_block_shape();
    constexpr int bytes=Kernel::SharedStorageSize;
    auto f=(CUfunction)vd_function;
    CUresult e=cuFuncSetAttribute(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,bytes);if(e)return 1000+e;
    CUlaunchConfig config={};config.gridDimX=grid.x;config.gridDimY=grid.y;config.gridDimZ=grid.z;
    config.blockDimX=block.x;config.blockDimY=block.y;config.blockDimZ=block.z;
    config.sharedMemBytes=bytes;config.hStream=(CUstream)stream;
    CUlaunchAttribute attr={};
    if(CX*CY>1){attr.id=CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;attr.value.clusterDim.x=CX;attr.value.clusterDim.y=CY;attr.value.clusterDim.z=1;config.attrs=&attr;config.numAttrs=1;}
    void* args[]={&params};e=cuLaunchKernelEx(&config,f,args,nullptr);return e?1000+e:0;'''+host[b:]
host=host.replace('namespace vd {','static thread_local void* vd_function=nullptr;\nnamespace vd {',1)
host=host.replace('int launch){\n#define CALL', 'int launch,void* function){\nvd_function=function;\n#define CALL')
(HERE/'cutlass_host.cpp').write_text(host)

# The host bridge is built in the same pinned image and linked only to its
# ordinary CUDA/C++/c10 runtime. Embedding it avoids repeating template compilation.
host_path=root/('muse_vdense_host_'+hashlib.sha256(host.encode()).hexdigest()[:12])/('muse_vdense_host_'+hashlib.sha256(host.encode()).hexdigest()[:12]+'.so')
if host_path.exists():
 host_bytes=host_path.read_bytes()
 (HERE/'host_binary.py.inc').write_text('VD_HOST_BINARY='+repr(base64.b64encode(zlib.compress(host_bytes,9)).decode())+'\nVD_HOST_SHA256='+repr(hashlib.sha256(host_bytes).hexdigest())+'\n')
 (HERE/'host_binary.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(host.encode()).hexdigest(),binary_sha256=hashlib.sha256(host_bytes).hexdigest(),bytes=len(host_bytes),platform='judge pinned image Linux x86_64 CUDA13; no computed inputs in binary'),indent=2)+'\n')
