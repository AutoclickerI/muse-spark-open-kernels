"""Cross-compile the actual asynchronous bodies and retain register/spill proof."""
import importlib.util,hashlib,json,re,subprocess,sysconfig
from pathlib import Path
from torch.utils import cpp_extension
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('da_pipeline_compile',H/'pipeline_source.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);ns=m.da_vendor();instances=[(bn,64,1,2,2,c,2) for bn in (64,128,256) for c in (False,True)];rows=[]
flags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-DTORCH_EXTENSION_NAME=da_pipe','-Xptxas=-v','-isystem',sysconfig.get_paths()['include']];flags += [a for p in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',p)]
for piped in (False,True):
    src=m.da_pipeline_source(ns,instances,piped);p=H/'native'/('pipeline_'+str(piped)+'.cu');p.write_text(src);run=subprocess.run(['nvcc',*flags,'-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=run.stdout+run.stderr;p.with_suffix('.log').write_text(log);assert run.returncode==0,log[-4000:];spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log);row=dict(pipeline=piped,source_sha256=hashlib.sha256(src.encode()).hexdigest(),registers=list(map(int,re.findall(r'Used (\d+) registers',log))),spills=spills);rows.append(row);print(row,flush=True)
(H/'pipeline_native_validation.json').write_text(json.dumps(dict(passed=True,units=rows),indent=2)+'\n')
