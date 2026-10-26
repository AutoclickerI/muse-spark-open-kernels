"""Compile exact new Hopper kernels without executing them on sm120."""
import concurrent.futures,hashlib,json,subprocess
from pathlib import Path
from torch.utils.cpp_extension import include_paths
H=Path(__file__).resolve().parent;O=H/'out'
def compile_one(name):
 p=O/(name+'.cu');out=O/(name+'.cubin')
 flags=['-O3','-std=c++20','-lineinfo','-DRSW=4','-DPRS=4','-DDRS=4','-DRBF=5','-gencode=arch=compute_90a,code=sm_90a'] if name.startswith('lee_') else ['-O3','-std=c++20','--expt-relaxed-constexpr','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']
 cmd=['/usr/local/cuda/bin/nvcc',*flags,*['-I'+x for x in include_paths('cuda')],'-I/usr/include/python3.12','-cubin',str(p),'-o',str(out),'-Xptxas=-v']
 r=subprocess.run(cmd,capture_output=True,text=True);(O/(name+'_compile.log')).write_text(r.stdout+r.stderr)
 return dict(name=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),ok=r.returncode==0,tail=(r.stdout+r.stderr)[-1600:])
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(compile_one,['lee_D','lee_V','lee_F','native_cs_vector','native_partial_vector']))
(H/'native_compile.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2));assert all(r['ok'] for r in rows)
