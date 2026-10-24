"""Run the entire measurement payload with the public, real-format loaded bank."""
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PREFILL = HERE.parents[2]/'muse_prefill_layer'
sys.path.insert(0,str(PREFILL))
spec = importlib.util.spec_from_file_location('prefill_public_reference',PREFILL/'reference.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
spec = importlib.util.spec_from_file_location('verify_lab',HERE.parent/'verify_streaming_lab.py')
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)
vs.VS_TRACE = True
weights = ref.generate_weights(1234)
started = time.monotonic()
lines = vs.lab(started+270,weights)
print('\n'.join(lines),flush=True)
assert len([l for l in lines if l.startswith('M=')])==12,lines[-3:]
assert not any(l.startswith(('REJECT','INCOMPLETE')) for l in lines),lines[-3:]
assert any('rejected=0' in l for l in lines if l.startswith('WEIGHT_WIDE_SUMMARY'))
(HERE/'lab_validation.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-started,lines=lines),indent=2)+'\n')
