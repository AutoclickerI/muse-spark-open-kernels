"""Pack native and grouped-point mechanisms with one frozen S15 source blob."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
s=(H/'p10_decoder_followup.py').read_text();needle="    return lines+[f'END complete={done}/10 elapsed={time.monotonic()-begin:.1f}s']"
assert s.count(needle)==1
s=s.replace(needle,needle.replace('return lines+','return (lines+')+', (m,weights,roles,ctrl,banks))')
s+='\n_native_decoder_lab=lab\nBLOCK_SOURCE='+repr((H/'block_delta.py.inc').read_text())+'\n'+(H/'point_lab.py.inc').read_text()
s+='''
def lab(deadline,weights,state=None):
    import traceback
    result=_native_decoder_lab(deadline,weights,state)
    if not isinstance(result,tuple):return result
    lines,context=result
    try:return lines+point_lab(deadline,context)
    except Exception as error:return lines+[f'POINT_ERROR {type(error).__name__}: {error}']+traceback.format_exc().splitlines()[-8:]
'''
compile(s,'<combined coding batch>','exec');assert len(s.encode())<1000000
(H/'p10_coding_structural.py').write_text(s)
(H/'combined_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),native_cells=10,point_cells=20,seeds=8),indent=2)+'\n');print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
