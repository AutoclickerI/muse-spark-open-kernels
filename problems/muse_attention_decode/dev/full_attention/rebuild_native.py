"""O3 sm90a instruction/resources proof before launching a structural arm."""
import concurrent.futures,hashlib,json,re,subprocess,sysconfig
from pathlib import Path
from torch.utils import cpp_extension
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--dir',default='rebuild');ap.add_argument('names',nargs='*');args=ap.parse_args()
H=Path(__file__).resolve().parent;folder=H/args.dir;out=folder/'native';out.mkdir(parents=True,exist_ok=True)
flags=['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas=-v','-DTORCH_EXTENSION_NAME=p5b_rebuild','-DTORCH_API_INCLUDE_EXTENSION_H','-isystem',sysconfig.get_paths()['include']]
for p in cpp_extension.include_paths(device_type='cuda'):flags+=['-isystem',p]
def compile_one(name):
 src=folder/(name+'.cu');obj=out/(name+'.o');p=subprocess.run([*flags,'-c',str(src),'-o',str(obj)],text=True,capture_output=True);log=p.stdout+p.stderr;(out/(name+'.log')).write_text(log)
 row=dict(name=name,cuda_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),ok=p.returncode==0)
 if p.returncode:row['error']=log[-6000:];return row
 sass=subprocess.run(['cuobjdump','--dump-sass',str(obj)],text=True,capture_output=True,check=True).stdout;(out/(name+'.sass')).write_text(sass)
 row.update(registers=re.findall(r'Used (\d+) registers',log),stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log),c75=re.findall(r'[^\n]*(?:C751[0-9]|C750[0-9])[^\n]*',log),hgmma=sum('QGMMA' in l or 'HGMMA' in l for l in sass.splitlines()),setmaxnreg=[l.strip() for l in sass.splitlines() if 'SETMAX' in l],m64n8k32=[l.strip() for l in sass.splitlines() if 'GMMA.64x8x32' in l])
 return row
names=args.names or [x.stem for x in folder.glob('*.cu') if x.stem in ['ready','reg','parity','kmajor','ready_elide']]
with concurrent.futures.ThreadPoolExecutor(3) as pool:rows=list(pool.map(compile_one,names))
(folder/('native'+('_'+'_'.join(names) if len(names)<4 else '')+'.json')).write_text(json.dumps(rows,indent=2)+'\n');print(rows,flush=True)
assert all(r['ok'] for r in rows)
