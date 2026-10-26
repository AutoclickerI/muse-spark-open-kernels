"""Production W applies only the measured O1/O8 function swaps; no pool/metadata changes."""
import ast,base64,difflib,hashlib,json,lzma
from pathlib import Path
from build import build,W_CODE
H=Path(__file__).resolve().parent

def integration():
    return W_CODE+"""
# Base preparation uses the original launchers at every role. Build O-only
# launchers afterward, then retain the incumbent's exact packed allocations.
_SC_W_PREWAIT_PORT=_port_cuda
_port_cuda=_sc_old_port
_sc_w_base_prepare=prepare
def prepare(weights):
    global _port_cuda
    state=_sc_w_base_prepare(weights)
    _port_cuda=_SC_W_PREWAIT_PORT
    try:prepared=nf_prepare(weights,{(1,'o'),(8,'o')})
    finally:_port_cuda=_sc_old_port
    for key,new in prepared['plans'].items():
        old=state['nf_port']['plans'][key]
        assert old['cfg']==new['cfg'] and old['idx']==new['idx']
        assert old['wp'].shape==new['wp'].shape and old['esc'].shape==new['esc'].shape
        state['nf_port']['plans'][key]=dict(old,fn=new['fn'])
    return state
"""

def main():
    base=build()['A'];inc=integration();source=base+'\n'+inc
    compile(source,'p7-W-production','exec');tag=hashlib.sha256(source.encode()).hexdigest()
    (H/'W_integration.py.inc').write_text(inc)
    (H/'W.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),source.splitlines(True),fromfile='a/submission.py',tofile='b/submission.py')))
    # Retain the original MIT notice in the single-file packed candidate.
    shipped=(H.parents[3]/'problems/muse_decode_layer/cand_r3_best7_s17_final.py').read_text()
    head='\n'.join(l.rstrip() for l in shipped.splitlines()[:ast.parse(shipped).body[0].lineno-1])+'\n';bundle=base64.b85encode(lzma.compress(source.encode(),preset=9)).decode()
    packed=head+'import torch,base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode('+repr(bundle)+')).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=='+repr(tag)+'\n_file="<p7-W-'+tag[:12]+'>"\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\nexec(compile(_source,_file,"exec"),globals())\n'
    (H/'W_candidate.py').write_text(packed)
    (H/'winner_manifest.json').write_text(json.dumps(dict(arm='W',expanded_sha256=tag,packed_sha256=hashlib.sha256(packed.encode()).hexdigest(),integration_sha256=hashlib.sha256(inc.encode()).hexdigest(),patch_sha256=hashlib.sha256((H/'W.patch').read_bytes()).hexdigest(),base_sha256=hashlib.sha256(base.encode()).hexdigest(),measured_recipe='prepare onlyO1/O8 native functions, replace fn in original NF plans; original wp/esc and all other role functions retained',reproduce='python3 '+str(H.relative_to(H.parents[3]))+'/build_winner.py'),indent=2)+'\n')
    print(tag)
if __name__=='__main__':main()
