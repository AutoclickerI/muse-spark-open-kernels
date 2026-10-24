"""Build a separate route/verify-glue experiment without changing the sent small-M lab."""
from pathlib import Path
import hashlib
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
cell='RG_CUDA='+repr((HERE/'route_moe.cu').read_text())+'\nRG_CPP='+repr((HERE/'route_moe.cpp').read_text())+'\n'
old=HERE.parent/'fused_glue'
cell+='FG_CUDA='+repr((HERE/'projection.cu').read_text())+'\n'+(old/'runtime.py.inc').read_text()
cell+='\n'+(HERE/'runtime.py.inc').read_text()
(HERE/'dropin.py').write_text(cell)
(HERE/'best7_route_glue_cell.py').write_text(cell+'\n'+(HERE/'adapter.py.inc').read_text())
if (HERE/'driver.py.inc').exists():
    helpers=(HERE.parent/'paired_lab/driver.py.inc').read_text().split('\ndef lab(deadline,weights,host_state):',1)[0]
    shared=(old/'driver.py.inc').read_text().split('\ndef lab(deadline,weights,host_state):',1)[0]
    shared=shared[:shared.index('\ndef fg_setup')]+shared[shared.index('\ndef fg_route_gate'):]
    body='BEST7_SOURCE='+repr((ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text())+'\n'+cell+'\n'+helpers+'\n'+shared+'\n'+(HERE/'driver.py.inc').read_text()
    sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
    assert not rejected_words(body),rejected_words(body)
    (ROOT/'prof/lab/best7_route_verify_glue.py').write_text(body)
    print('payload',hashlib.sha256(body.encode()).hexdigest())
print('dropin',hashlib.sha256(cell.encode()).hexdigest())
