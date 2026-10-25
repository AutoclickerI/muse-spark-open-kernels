"""Freeze the small-M NT1 lab: M=1 globaltimer timeline + flag-templated variants paired against the admitted INC."""
import hashlib
import re
import sys
from pathlib import Path

from build_moe_nt1_limit import literal
from build_moe_small_debug import select

ROOT=Path(__file__).resolve().parents[2]
DEV=ROOT/'problems/muse_decode_layer/dev/nt1_timeline'
sys.path.insert(0,str(DEV))
from build_var import var_source  # noqa: E402


def build():
    old=(ROOT/'prof/lab/moe_nt1_limit.py').read_text();frozen=literal(old,'NT1_SOURCES')
    inc=frozen['lop3'];assert inc.strip()==(DEV/'inc.cu').read_text().strip()
    var=var_source();(DEV/'var.cu').write_text(var)
    a=old.index('NT1_SOURCES=');b=old.index('\nH =',a)
    body=old[:a]+'NT1_SOURCES='+repr({'INC':inc})+old[b:old.index('\nimport hashlib\nimport statistics\nimport time\n')]
    body+='\nVAR_CUDA='+repr(var)
    body+='\nACTIVITY_CPP='+repr((DEV/'activity.cpp.inc').read_text())
    body+='\nCONTROL_CUDA='+repr((DEV/'controls.cu').read_text())
    body+='\n'+(DEV/'runtime.py.inc').read_text()
    body+='\n'+(DEV/'var_runtime.py.inc').read_text()
    base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text()
    router=select(base,{'_pdl_start','_pdl_wait','_route_fast'})
    router=router.replace('def _route_fast(', 'def timeline_route(')
    router=re.sub(r'(?m)^def ', '@triton.jit\ndef ',router)
    body+='\nimport triton\nimport triton.language as tl\n'+router.rstrip()+'\n'
    body+='\n'+(DEV/'router_runtime.py.inc').read_text()
    body+=select((ROOT/'problems/muse_decode_layer/dev/nt1_branches/fixed_runtime.py.inc').read_text(),{'nt1_fp32'})
    for path in ('diagnose.py.inc','lab.py.inc'):body+='\n'+(DEV/path).read_text()
    out=ROOT/'prof/lab/moe_nt1_timeline.py';out.write_text(body)
    print(out.relative_to(ROOT),hashlib.sha256(body.encode()).hexdigest())


if __name__=='__main__':build()
