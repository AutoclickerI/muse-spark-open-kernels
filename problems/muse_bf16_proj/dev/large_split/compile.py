"""Compile mechanism bodies and inspect actual register/stack/issue instructions."""
import importlib.util,hashlib,json,re,subprocess,sysconfig
from pathlib import Path
from torch.utils import cpp_extension
import source
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('ls_compile',H/'generated/source.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);ns=m._nf_vendor();out=H/'generated/native';out.mkdir(parents=True,exist_ok=True)
flags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-DTORCH_EXTENSION_NAME=ls_compile','-Xptxas=-v','-isystem',sysconfig.get_paths()['include']];flags += [a for p in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',p)]
rows=[]
for mode in source.MODES:
 s=source.source(ns,mode);p=out/(mode+'.cu');p.write_text(s);run=subprocess.run(['nvcc',*flags,'-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=run.stdout+run.stderr;p.with_suffix('.log').write_text(log);assert run.returncode==0,log[-5000:];sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass);row=dict(mode=mode,source_sha256=hashlib.sha256(s.encode()).hexdigest(),registers=list(map(int,re.findall(r'Used (\d+) registers',log))),stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log),HGMMA=sass.count('HGMMA'),TMA=sass.count('UTMALDG'),global_loads=sass.count('LDG.'),global_stores=sass.count('STG.'));rows.append(row);print(row,flush=True)
 assert row['HGMMA']==0 if 'mathfree' in mode else row['HGMMA']>0
 assert row['global_loads']>=36,('weight traffic lost',row)
(H/'native_validation.json').write_text(json.dumps(dict(passed=True,units=rows),indent=2)+'\n')
