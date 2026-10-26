"""Compile the exact staged kernels and S22 native O128/O256 specializations."""
import concurrent.futures,hashlib,importlib.util,json,re,subprocess,sysconfig
from pathlib import Path
from torch.utils.cpp_extension import include_paths
H=Path(__file__).resolve().parent;R=H.parents[4];O=H/'out'
sp=importlib.util.spec_from_file_location('night_s22_native',R/'problems/muse_decode_layer/cand_r3_best7_s22_final.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
units=[('stage',(O/'stage.cu').read_text(),['-DRSW=4','-DPRS=4','-DDRS=4','-DRBF=5'])]
ns=m._nf_vendor()
class Capture:
 def __getattr__(self,name):return lambda *a,**kw:None
def capture(name,cpp_sources='',cuda_sources='',**kw):units.append((re.sub(r'[^a-zA-Z0-9_]','_',name),cuda_sources,[]));return Capture()
ns['load_inline']=capture;ns['OLAYER_CUDA']=m._nf_pdl_wide(ns['OLAYER_CUDA']);ns['build_olayer']([(128,128,4,3,1,1)])
ns['tt_v10_tune_RL_CUDA']=m._nf_pdl_wide(ns['tt_v10_tune_RL_CUDA']);ns['tt_v10_tune_build_rl']([(256,64,5,7,1,1,1)])
def one(unit):
 name,source,flags=unit;p=O/(name+'.cu');p.write_text(source)
 cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-std=c++20','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a','-DTORCH_EXTENSION_NAME=p5b_night_native','-DTORCH_API_INCLUDE_EXTENSION_H',*flags,*['-I'+x for x in include_paths('cuda')],'-I'+sysconfig.get_paths()['include'],'-c',str(p),'-o',str(O/(name+'.o')),'-Xptxas=-v']
 run=subprocess.run(cmd,capture_output=True,text=True);log=run.stdout+run.stderr;(O/(name+'_compile.log')).write_text(log)
 row=dict(name=name,source_sha256=hashlib.sha256(source.encode()).hexdigest(),passed=run.returncode==0,registers=re.findall(r'Used (\d+) registers',log),stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log))
 if run.returncode:row['error']=log[-2600:];return row
 sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(O/(name+'.o'))],text=True);(O/(name+'.sass')).write_text(sass)
 row['entry_release']=sass.count('PREEXIT');row['dependency_wait']=sass.count('ACQBULK');assert row['entry_release'] and row['dependency_wait']
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,units))
(H/'native_compile.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2));assert len(rows)==3 and all(r['passed'] for r in rows)
