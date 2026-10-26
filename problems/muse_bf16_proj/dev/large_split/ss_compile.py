"""Compile all shared/shared WGMMA token-warpgroups on sm90a."""
import hashlib,json,re,subprocess,sysconfig
from pathlib import Path
from torch.utils import cpp_extension
import ss_source
H=Path(__file__).resolve().parent;p=H/'generated/native/ss.cu';p.parent.mkdir(parents=True,exist_ok=True);s=ss_source.source();p.write_text(s)
flags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-DTORCH_EXTENSION_NAME=ls_ss','-Xptxas=-v','-isystem',sysconfig.get_paths()['include']];flags += [a for p in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',p)]
r=subprocess.run(['nvcc',*flags,'-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=r.stdout+r.stderr;p.with_suffix('.log').write_text(log);assert r.returncode==0,log[-5000:];regs=list(map(int,re.findall(r'Used (\d+) registers',log)));spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log);d=dict(passed=True,source_sha256=hashlib.sha256(s.encode()).hexdigest(),configs=ss_source.CONFIGS,registers=regs,stack_spills=spills);(H/'ss_native_validation.json').write_text(json.dumps(d,indent=2)+'\n');print(d)
