"""Fit the paired lab on one decode check: use frozen S11 as host and reuse its actual prepared state."""
import hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import h200_lab
base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('acda6c62')
s=(ROOT/'prof/lab/full128_timeline.py').read_text().replace('def lab(deadline,weights,state=None):','def p5_pair(deadline,weights,state=None):')
s+='''
class _P5LiveS11:
    def __init__(self,ns):self.ns=ns
    @property
    def __dict__(self):return self.ns
    def __getattr__(self,key):return self.ns[key]

def lab(deadline,weights,state):
    global TL_EXISTING_HOST
    # The host is exactly frozen S11. Reuse its selected state, avoiding another ~70 s setup.
    # The graph wrappers mutate the real globals dictionary, preserving original graph-node attribution.
    TL_EXISTING_HOST=(_P5LiveS11(kernel_fn.__globals__),state)
    return p5_pair(deadline,weights,state)
'''
compile(s,'full128_decode_pair.py','exec');p=ROOT/'prof/lab/full128_decode_pair.py';p.write_text(s);tag=hashlib.sha256(s.encode()).hexdigest()[:16]
wrapped=base+'\n\n'+h200_lab.WRAPPER.format(names='full128_decode_pair '+tag,base='frozen S11 acda6c62',payloads=[('full128_decode_pair',tag,s)],budget=285,limit=9000)
out=ROOT/'problems/muse_decode_layer/lab/lab_p5_full128_fixed_s11.py';compile(wrapped,str(out),'exec');out.write_text(wrapped);print(out,hashlib.sha256(wrapped.encode()).hexdigest(),len(wrapped))
