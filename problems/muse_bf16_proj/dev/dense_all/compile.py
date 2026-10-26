"""Compile every changed Hopper source; save resources and wait/load ordering."""
import importlib.util,hashlib,json,re,subprocess,sysconfig
from pathlib import Path
from torch.utils import cpp_extension
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('da_compile',H/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ns=m.da_vendor();units=[]
def capture(name,cpp_sources='',cuda_sources='',**kw):
    units.append((name,cuda_sources));return None
m.load_inline=capture
for e in (False,True):
    for gate in (False,True):m.da_build_small(ns,e,gate)
insts=list(dict.fromkeys((bn or t,64,1,pf,2,True,2) for t in (128,256) for bn,s,pf,e in m.DA_WIDE))
for early in (False,True):units.append(('wide_'+str(early),m.da_wide_source(ns,insts,early)))
folder=H/'native';folder.mkdir(exist_ok=True)
flags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-DTORCH_EXTENSION_NAME=da_compile','-DTORCH_API_INCLUDE_EXTENSION_H','-Xptxas=-v','-isystem',sysconfig.get_paths()['include']]
flags += [a for path in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',path)]
rows=[]
for name,source in units:
    p=folder/(name+'.cu');p.write_text(source);run=subprocess.run(['nvcc',*flags,'-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=run.stdout+run.stderr;p.with_suffix('.log').write_text(log)
    row=dict(name=name,source_sha256=hashlib.sha256(source.encode()).hexdigest(),passed=run.returncode==0,registers=list(map(int,re.findall(r'Used (\d+) registers',log))),stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log));rows.append(row);print(row,flush=True)
    if run.returncode:raise RuntimeError(log[-6000:])
(H/'native_validation.json').write_text(json.dumps(dict(passed=True,units=rows),indent=2)+'\n')
