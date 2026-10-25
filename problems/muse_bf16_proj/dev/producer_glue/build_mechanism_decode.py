"""Use S13 final as the unchanged decode lab host when component quota is spent."""
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO/'tools'))
import h200_lab

base=(REPO/'problems/muse_decode_layer/cand_r3_best7_s13_final.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest().startswith('958f1258')
embedded=[h200_lab.embed(HERE/'lab_mechanism.py')]
wrapped=base.rstrip()+'\n\n'+h200_lab.WRAPPER.format(names='P7 glue mechanism',base='S13 958f1258',
    payloads=embedded,budget=300-h200_lab.MARGIN_S,limit=h200_lab.LIMIT)
compile(wrapped,'mechanism_decode','exec')
out=HERE/'mechanism_decode_trial.py';out.write_text(wrapped)
print(out,len(wrapped),hashlib.sha256(wrapped.encode()).hexdigest())
