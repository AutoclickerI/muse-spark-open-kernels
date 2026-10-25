"""Compile actual SM90a route code and prove selected GEMMs do not load STATE/INV."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

H=Path(__file__).resolve().parent;R=H.parents[4]


def main():
    payload=R/'prof/lab/moe_route_init_m256.py';tree=ast.parse(payload.read_text());cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='FX_CUDA' for t in n.targets))
    src=H/'audit.cu';src.write_text(cuda);ptx=src.with_suffix('.ptx')
    p=subprocess.run(['nvcc','--ptx','-O3','-std=c++20','-arch=sm_90a','-lineinfo',str(src),'-o',str(ptx)],capture_output=True,text=True);(H/'native.log').write_text(p.stdout+p.stderr);p.check_returncode()
    text=ptx.read_text();rows=[];other=[];route=None
    for entry in text.split('.visible .entry ')[1:]:
        name=entry.split('(',1)[0].strip();body=entry[entry.index('\n{'):];params=re.findall(r'\.param\s+\.\w+\s+(\S+?)(?:,|\n)',entry[:entry.index('\n{')])
        if 'fusedILi' in name:
            assert len(params)==24,(name,len(params))
            loaded={index:bool(re.search(r'ld\.param[^;]*\['+re.escape(params[index])+r'\]',body)) for index in (17,18)}
            if re.search(r'fusedILi(?:16|32)ELi128ELi4ELi1ELb[01]ELi1',name):
                assert not any(loaded.values()),(name,loaded)
                rows.append(dict(name=name,INV_loaded=False,STATE_loaded=False))
            else:other.append(dict(name=name,INV_loaded=loaded[17],STATE_loaded=loaded[18]))
        if 'mrinit10route_live' in name:
            for index in (7,8,12):assert not re.search(r'ld\.param[^;]*\['+re.escape(params[index])+r'\]',body),(name,index)
            route=dict(name=name,dead_state_loaded=False,dead_out_loaded=False,dead_clear_loaded=False,signature_params=len(params))
    assert route and len(rows)>=8 and any(r['STATE_loaded'] for r in other)
    c=H/'route.cu';cube=c.with_suffix('.cubin');p=subprocess.run(['nvcc','--cubin','-O3','-std=c++20','-arch=sm_90a','-Xptxas=-v',str(c),'-o',str(cube)],capture_output=True,text=True);(H/'route.ptxas.log').write_text(p.stdout+p.stderr);p.check_returncode()
    sass=subprocess.run(['cuobjdump','--dump-sass',str(cube)],capture_output=True,text=True,check=True).stdout;c.with_suffix('.sass').write_text(sass);assert not re.search(r'\b(?:LDL|STL)[.\s]',sass)
    out=dict(payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),selected_mode1_split1_GEMMs=rows,other_modes_kept=other,route=route,route_local_memory_instructions=0,full_cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest())
    (H/'native.json').write_text(json.dumps(out,indent=2)+'\n');print('ROUTE_NATIVE_PASS',len(rows),route,flush=True)


if __name__=='__main__':main()
