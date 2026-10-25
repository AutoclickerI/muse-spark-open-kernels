"""Build final-S11 host with the direct-partials port and an exact S11 control for paired H200 runs."""
import ast
import hashlib
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import h200_lab
from okc_run import rejected_words
host=(HERE/'out/s11_partial_port.py').read_text()
original=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text()
lines=original.splitlines()
node=next(n for n in ast.parse(original).body if getattr(n,'name','')=='_best7_attention')
control='\n'.join(lines[node.lineno-1:node.end_lineno])
helpers=(ROOT/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc').read_text()
helpers=helpers[:helpers.index('def lab(deadline, weights, state):')].replace('__NT1_CUDA__',"''").replace('__NT1_CPP__',"''")
body=helpers+'\nP5_CONTROL_FUNCTION = '+repr(control)+'\n'+(HERE/'layer_lab.py.inc').read_text()
payload=ROOT/'prof/lab/attention_p5_layer.py';payload.write_text(body)
sha=hashlib.sha256(body.encode()).hexdigest()[:16]
src=host+'\n\n'+h200_lab.WRAPPER.format(names='attention_p5_layer '+sha,base='final S11 acda6c62 + partial port',payloads=[('attention_p5_layer',sha,body)],budget=285,limit=9000)
assert not rejected_words(src),rejected_words(src)
compile(src,'p5_layer','exec')
out=ROOT/'problems/muse_decode_layer/lab/lab_attention_p5_layer.py';out.write_text(src)
print(out,hashlib.sha256(src.encode()).hexdigest())
