"""Compile the actual frozen/clocked sources for sm90a; reject WGMMA serialization."""
import hashlib
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

H = Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('sources',nargs='+')
ap.add_argument('-D',dest='defines',action='append',default=[])
a=ap.parse_args()
rows = []
with tempfile.TemporaryDirectory(prefix='full-attn-native-') as tmp:
    for f in a.sources:
        flags=['-D'+d for d in a.defines]
        p = H / f
        r = subprocess.run(['nvcc', '-std=c++20', '-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a',
            '-Xptxas=-v', '-c', str(p), '-o', str(Path(tmp) / (f + '.o'))] + flags,
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert not re.search(r'C75\d+', r.stderr), r.stderr
        for stores, loads in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads', r.stderr):
            assert int(stores) == int(loads) == 0, r.stderr
        rows.append(dict(file=f, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         ok=True, flags=flags, log=r.stderr))
(H / 'native_admission.json').write_text(json.dumps(rows, indent=2) + '\n')
print('Native admission passes: no spills or C75')
