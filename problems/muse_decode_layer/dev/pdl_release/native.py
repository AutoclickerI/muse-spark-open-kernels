"""Optimized sm90a placement evidence for the four changed CUDA sources."""
import ast
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sysconfig
import sys
from pathlib import Path
from torch.utils import cpp_extension
H=Path(__file__).resolve().parent
s=(H.parents[1]/'cand_r3_best7_s15e.py').read_text()
values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(s).body
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and isinstance(n.value,ast.Constant)}
out=H/'out/native';out.mkdir(exist_ok=True)
base=['nvcc','-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a',
      '--expt-relaxed-constexpr','-Xptxas=-v','-DTORCH_EXTENSION_NAME=pdl_release_check',
      '-DTORCH_API_INCLUDE_EXTENSION_H','-isystem',sysconfig.get_paths()['include']]
for p in cpp_extension.include_paths(device_type='cuda'):base+=['-isystem',p]


def one(name):
    cu=out/(name+'.cu');cu.write_text(values[name]);obj=cu.with_suffix('.o')
    p=subprocess.run([*base,'-c',str(cu),'-o',str(obj)],capture_output=True,text=True) if '--parse-only' not in sys.argv else subprocess.CompletedProcess([],0,'',next(r['compiler'] for r in json.loads((H/'optimized_native.json').read_text()) if r['name']==name))
    r=dict(name=name,sha256=hashlib.sha256(cu.read_bytes()).hexdigest(),ok=p.returncode==0,
           compiler=p.stdout+p.stderr,functions=[])
    if p.returncode:return r
    sass=subprocess.run(['cuobjdump','--dump-sass',str(obj)],capture_output=True,text=True,check=True).stdout
    for item in re.split(r'\n\s*Function : ',sass)[1:]:
        lines=item.splitlines();ops=[l.strip() for l in lines if '/*' in l and ';' in l]
        triggers=[i for i,l in enumerate(ops) if 'PREEXIT' in l]
        waits=[i for i,l in enumerate(ops) if 'ACQBULK' in l]
        loads=[i for i,l in enumerate(ops) if re.search(r'\b(LDG|LDGSTS|UTMALDG)',l)]
        if triggers or waits:r['functions'].append(dict(symbol=lines[0],instructions=len(ops),
            trigger_instruction=triggers,wait_instruction=waits,first_global_load=loads[:1],
            snippets=[ops[i] for i in sorted(set(triggers+waits+loads[:1]))]))
    (out/(name+'.sass')).write_text(sass)
    return r


with concurrent.futures.ThreadPoolExecutor(4) as pool:
    rows=list(pool.map(one,('ATTN_PORT_CUDA_SRC','SMALLM_VAR_CUDA','NF_CUDA','V9_CUDA_SRC')))
(H/'optimized_native.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['name'],r['ok'],len(r['functions']),r['compiler'][-300:] if not r['ok'] else '')
assert all(r['ok'] and r['functions'] for r in rows)
