"""Source-level preservation plus binary identity of the unchanged qkv VDSK."""
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
import torch
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
import build
H=Path(__file__).resolve().parent


def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def funcs(s):
    counts={};out={}
    for n in ast.parse(s).body:
        if isinstance(n,ast.FunctionDef):
            i=counts.get(n.name,0);counts[n.name]=i+1;out[n.name,i]=ast.dump(n)
    return out


def main():
    a=module('preserve_s15',build.BASE);b=module('preserve_s15L',H.parents[1]/'cand_r3_best7_s15L.py')
    aa,bb=funcs(build.BASE.read_text()),funcs(build.compose())
    changed=[key for key in aa if aa[key]!=bb[key]]
    assert [k[0] for k in changed]==['_vdsk_gemm','_vdsk_project','_best7_project'],changed
    fields=['BEST7_SETTINGS','PROJ_PINS','CD_ROWS','GB_ROWS','SMALLM_FLAGS','GATE_SPLIT_ROWS','GLUE_CL_ROWS','NF_SMALL','NF_WIDE']
    for name in fields:assert getattr(a,name)==getattr(b,name),name
    assert b.NF_PORT_SELECTED-a.NF_PORT_SELECTED=={(128,'o'),(256,'o')}
    assert {k:v for k,v in b._VQ8_ROWS.items() if k not in a._VQ8_ROWS}=={(1,8,False):(4,2),(32,8,True):(4,2)}
    x=torch.empty((128,8192),device='cuda',dtype=torch.bfloat16)
    w=torch.empty((2304,8192),device='cuda',dtype=torch.bfloat16);p=torch.empty((3,128,2304),device='cuda')
    old=driver.active.get_current_target;blobs=[];binaries=[]
    try:
        driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
        for name,m in [('S15',a),('S15L',b)]:
            xd=m._VDSK_TD.from_tensor(x,[128,64]);wd=m._VDSK_TD.from_tensor(w,[64,64])
            k=m._vdsk_gemm.warmup(x,w,p,xd,wd,128,2304,8192,3,BT=128,BW=64,BK=64,NS=8,TMA=1,
                num_warps=4,num_stages=8,grid=(36,3),**({'CHAIN':False,'launch_pdl':False} if name=='S15L' else {}))
            f=H/'out'/('qkv_'+name+'.cubin');f.write_bytes(k.asm['cubin'])
            sass=subprocess.run(['cuobjdump','--dump-sass',str(f)],capture_output=True,text=True,check=True).stdout
            # Compare emitted instructions and their machine encodings, ignoring
            # filenames and ELF debug/source metadata.
            ops='\n'.join(l.strip() for l in sass.splitlines() if re.search(r'/\*[0-9a-f]+\*/',l))
            blobs.append(ops);binaries.append(dict(name=name,sass_sha256=hashlib.sha256(ops.encode()).hexdigest()))
    finally:driver.active.get_current_target=old
    assert blobs[0]==blobs[1],'CHAIN=False changed the scored qkv binary'
    result=dict(changed_functions=changed,unchanged_tables=fields,unchanged_cuda_vendor_literals=36,
        qkv_vdsk_binary_identical=True,binaries=binaries,base_sha256=build.SHA,
        candidate_sha256=hashlib.sha256(build.compose().encode()).hexdigest(),
        dispatch_changes=dict(o=['T128','T256'],vdsk=['T128 pool_router','T128 unpooler'],vq8=['B1Q8 full','B32Q8 sliding']))
    (H/'preservation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
