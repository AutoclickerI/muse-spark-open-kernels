"""The same all-layer comparator with compact cooperative prep, fixed split counts, and phase clocks."""
import ast,hashlib
from pathlib import Path
import build_compact,build_clock
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(ROOT/'prof/lab/warp_sliding_timeline.py').read_text();tree=ast.parse(s);lines=s.splitlines(True)
for n in reversed(tree.body):
    if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('cs_CUDA','CS_CLOCK_CUDA'):
        key=n.targets[0].id;val=build_compact.source() if key=='cs_CUDA' else {False:build_clock.source(),True:build_clock.source(cuda=build_compact.source())}
        lines[n.lineno-1:n.end_lineno]=[key+'='+repr(val)+'\n']
s=''.join(lines)
a=s.index('_CS_CLOCK_MOD={}');b=s.index('def lab(deadline,weights,state=None):',a)
s=s[:a]+(HERE/'clock_runtime.py.inc').read_text()+'\n'+s[b:]
s=s.replace('32-thread m16 FP8 MMA + DSM, auto split','compact cooperative Q/K/V projection reduction, native WGMMA + DSM, auto split')
compile(s,'compact_sliding_timeline.py','exec');p=ROOT/'prof/lab/compact_sliding_timeline.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s))
