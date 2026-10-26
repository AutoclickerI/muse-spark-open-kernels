"""Reconstruct the frozen P12 control from tracked inputs, without side effects."""
import hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[4]
def projection_source():
    h=R/'problems/muse_decode_layer/dev/p12_components'
    source=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest().startswith('60254883')
    license=(R/'problems/muse_bf16_proj/dev/public_nf/LICENSE.NathenFernandes').read_text()
    notice='\n'.join('# '+line for line in ('Nathen Fernandes; github.com/NathenFernandes/muse-bf16-proj @ 86aff75; MIT\n'+license).splitlines())
    source='# okc: problem=muse_bf16_proj\n'+notice+'\n'+source+'\n'+(h/'projection.py.inc').read_text()+'\n'+(h/'projection_reduce.py.inc').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()=='2d5caebc4df2175c1aa22bfb7fde60ac42893145820d64d24f5b63cdf070a781'
    return source
