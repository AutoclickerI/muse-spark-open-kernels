"""Build the standalone best7 glue cell and one synthetic-dense / real-expert lab."""
from pathlib import Path
import hashlib
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
source=ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py'
cell='FG_CUDA='+repr((HERE/'kernel.cu').read_text())+'\n'+(HERE/'runtime.py.inc').read_text()
(HERE/'dropin.py').write_text(cell)
(HERE/'best7_glue_cell.py').write_text(cell+'\n'+(HERE/'adapter.py.inc').read_text())
if (HERE/'driver.py.inc').exists():
    helpers=(HERE.parent/'paired_lab/driver.py.inc').read_text().split('\ndef lab(deadline,weights,host_state):',1)[0]
    body='BEST7_SOURCE='+repr(source.read_text())+'\n'+cell+'\n'+helpers+'\n'+(HERE/'flip_gate.py.inc').read_text()+'\n'+(HERE/'driver.py.inc').read_text()
    sys.path.insert(0,str(ROOT/'tools'))
    from okc_run import rejected_words
    assert not rejected_words(body),rejected_words(body)
    (ROOT/'prof/lab/best7_fused_glue.py').write_text(body)
    print('payload',hashlib.sha256(body.encode()).hexdigest())
print('cell',hashlib.sha256(cell.encode()).hexdigest())
