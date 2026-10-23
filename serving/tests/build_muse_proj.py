"""Embed unchanged receipt skinny kernels and Lt ABI in the serving module.

Development tool only: the deployed module imports no problem/development file.
It changes only the generated CUDA literal in the lane's muse_proj.py.
"""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
receipt=ROOT/'problems/muse_bf16_proj/cand_fused_select.py'
text=receipt.read_text()
assert hashlib.sha256(text.encode()).hexdigest()=='10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9'
assign={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(text).body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('RAW_CUDA','INSTANCES')}
raw=assign['RAW_CUDA']
skin=raw[raw.index('namespace muse_proj {'):raw.index('namespace muse_pair {')]
host=ROOT/'problems/muse_bf16_proj/dev/lean_host.cuh'
s=host.read_text();prefix=s[:s.index('LEAN_DEVICE_SKINNY')]
launch=s[s.index('template<int NT'):s.index('static int lean_map')]
cases=[c for c in assign['INSTANCES'] if c[:4] in [(1,1,4,1),(1,2,4,1),(2,1,4,1),(2,2,4,1)]]
launch=launch.replace('LEAN_SKIN_CASES','\n'.join('  LSKIN('+','.join(map(str,c))+')' for c in cases))
lt=s[s.index('struct LeanLayout{'):]
cuda=prefix+skin+launch+lt
# Macro continuation is semantic whitespace. The serving patch disallows any
# added line >400 columns, including the native ABI's generated dispatch macro.
lines=[]
for line in cuda.splitlines():
    while len(line)>390:
        at=line.rfind(' ',0,365)
        assert at>0
        lines.append(line[:at]+' \\');line='  '+line[at+1:]
    lines.append(line)
cuda='\n'.join(lines)+'\n'
path=ROOT/'serving/sglang/python/sglang/srt/layers/muse_proj.py'
source=path.read_text().split('\n_CUDA_SOURCE = r"""',1)[0].rstrip()
source+='\n\n\n_CUDA_SOURCE = r"""'+cuda+'"""\n'
compile(source,str(path),'exec');path.write_text(source)
print('embedded',len(cases),'receipt skinny specializations; source bytes',len(source))
print('max line',max(map(len,source.splitlines())))
