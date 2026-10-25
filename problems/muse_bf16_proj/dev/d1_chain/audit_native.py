"""Compile and disassemble native sm90a sources for the glue ordering audit."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import torch
from torch.utils import cpp_extension

def load(path):
    spec=importlib.util.spec_from_file_location('pdl_audit_layer',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

def compile_one(item,folder):
    name,src=item;path=folder/(name+'.cu')
    reuse=path.exists() and path.read_text()==src and path.with_suffix('.o').exists()
    path.write_text(src)
    base=['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a',
          '--expt-relaxed-constexpr','-DTORCH_EXTENSION_NAME=pdl_audit',
          '-DTORCH_API_INCLUDE_EXTENSION_H','-D__CUDA_NO_HALF_OPERATORS__',
          '-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
          '-D__CUDA_NO_HALF2_OPERATORS__','-isystem',sysconfig.get_paths()['include']]
    base += [a for p in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',p)]
    result=subprocess.CompletedProcess([],0,'','') if reuse else subprocess.run([*base,'-c',str(path),'-o',str(path.with_suffix('.o'))],capture_output=True,text=True)
    if not reuse:path.with_suffix('.log').write_text(result.stdout+result.stderr)
    record=dict(name=name,sha256=hashlib.sha256(src.encode()).hexdigest(),ok=result.returncode==0)
    if result.returncode:
        record['error']=(result.stdout+result.stderr)[-2000:];print(record,flush=True);return record
    result=subprocess.run(['cuobjdump','--dump-sass',str(path.with_suffix('.o'))],capture_output=True,text=True,check=True)
    path.with_suffix('.sass').write_text(result.stdout)
    elf=folder/(name+'_elf');elf.mkdir(exist_ok=True)
    subprocess.run(['cuobjdump','--extract-elf','all',str(path.with_suffix('.o').resolve())],cwd=elf,capture_output=True,text=True,check=True)
    for cubin in elf.glob('*.cubin'):
        dis=subprocess.run(['nvdisasm','-g',str(cubin)],capture_output=True,text=True,check=True)
        cubin.with_suffix('.sass').write_text(dis.stdout)
    usage=subprocess.run(['cuobjdump','--dump-resource-usage',str(path.with_suffix('.o'))],capture_output=True,text=True,check=True).stdout
    path.with_suffix('.resources').write_text(usage)
    kernels=[]
    for block in re.split(r'Function : ',result.stdout)[1:]:
        name=block.splitlines()[0].strip();lines=block.splitlines();instructions=[];line=None
        for text in lines:
            if 'line ' in text:
                match=re.search(r'line (\d+)',text)
                if match:line=int(match[1])
            if re.search(r'/\*[0-9a-f]+\*/',text):instructions.append((text.strip(),line))
        waits=[i for i,(s,_) in enumerate(instructions) if 'ACQBULK' in s]
        releases=[i for i,(s,_) in enumerate(instructions) if 'PREEXIT' in s]
        if not waits and not releases:continue
        loads=[dict(index=i,line=line,sass=s) for i,(s,line) in enumerate(instructions)
               if re.search(r'\b(?:LDG|LDGSTS|CCTL)\b',s)]
        kernels.append(dict(symbol=name,instructions=len(instructions),waits=waits,releases=releases,
            independent_candidates_before_first_wait=[x for x in loads if waits and x['index']<waits[0]],
            load_source_lines_after_wait=sorted(set(x['line'] for x in loads if waits and x['index']>waits[0] and x['line'] is not None))))
    record['kernels']=kernels
    print(name if not kernels else f'{record["name"]}: {len(kernels)} PDL specializations',flush=True)
    return record
