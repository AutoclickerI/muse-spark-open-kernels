"""Compile the real Hopper kernel; reject spills and inspect native instructions."""
import importlib.util
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('vs_projection',HERE/'projection.py')
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)

with tempfile.TemporaryDirectory(prefix='verify_streaming_') as folder:
    source = Path(folder)/'kernel.cu'
    cubin = Path(folder)/'kernel.cubin'
    source.write_text(vs.VS_CUDA)
    build = subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_90a','-O3','-std=c++17','-lineinfo',
                            '-Xptxas=-v','--cubin',str(source),'-o',str(cubin)],capture_output=True,text=True)
    (HERE/'compile_sm90a.log').write_text(build.stdout+build.stderr)
    build.check_returncode()
    sass = subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(cubin)],
                          capture_output=True,text=True,check=True).stdout
    (HERE/'sm90a.sass.log').write_text(sass)
    functions = []
    for block in build.stderr.split('Compiling entry function ')[1:]:
        name = block.splitlines()[0]
        registers = int(re.search(r'Used (\d+) registers',block).group(1))
        spills = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',block)
        stores,loads = map(int,spills.groups())
        functions.append(dict(name=name,registers=registers,stores=stores,loads=loads))
        assert stores==loads==0,(name,stores,loads)
    assert len(functions)==len(vs.VS_CONFIGS),(len(functions),len(vs.VS_CONFIGS))
    counts = {n:len(re.findall(n,sass)) for n in ('HGMMA','UTMALDG','WARPGROUP','SYNCS','STG','LDG','ATOM')}
    assert counts['HGMMA'] and counts['UTMALDG'],counts
    result = dict(source_sha256=hashlib.sha256(vs.VS_CUDA.encode()).hexdigest(),
                  functions=functions,sass_counts=counts,no_spills=True,native_execution=False)
    (HERE/'compile_sm90a.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SM90A_AUDIT',json.dumps(result),flush=True)
