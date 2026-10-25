"""Official last-CTA candidate validation plus the corrected register-only N128 experiment in prepare."""
import hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import h200_lab
host=(HERE.parent/'cluster_sliding/out/s11_last_decode1.py').read_text()
assert hashlib.sha256(host.encode()).hexdigest().startswith('fead4a00')
s=(ROOT/'prof/lab/full128_timeline.py').read_text().replace('def lab(deadline,weights,state=None):','def p5_full_pair(deadline,weights,state=None):').replace('range(8))','range(16))',1)
s=s.replace('            for g in graphs.values():g.replay()\n            torch.cuda.synchronize()',"            for g in graphs.values():g.replay()\n            torch.cuda.synchronize()\n            assert all(bool(torch.isfinite(x).all()) for value in out.values() for x in value[:2]), ('nonfinite full replay',label,seed)")
s+='''
class _P5LiveS11:
    def __init__(self,ns):self.ns=ns
    @property
    def __dict__(self):return self.ns
    def __getattr__(self,key):return self.ns[key]

def lab(deadline,weights,state):
    global TL_EXISTING_HOST
    TL_EXISTING_HOST=(_P5LiveS11(kernel_fn.__globals__),state)
    return p5_full_pair(deadline,weights,state)
'''
compile(s,'full128_regmax_pair.py','exec');p=ROOT/'prof/lab/full128_regmax_pair.py';p.write_text(s);tag=hashlib.sha256(s.encode()).hexdigest()[:16]
out=host+'\n\n'+h200_lab.WRAPPER.format(names='full128_regmax_pair '+tag,base='S11 + last-CTA decode1 sliding fead4a00',payloads=[('full128_regmax_pair',tag,s)],budget=285,limit=9000)
p=ROOT/'problems/muse_decode_layer/lab/lab_p5_last_and_regmax.py';compile(out,str(p),'exec');p.write_text(out);print(p,hashlib.sha256(out.encode()).hexdigest(),len(out))
