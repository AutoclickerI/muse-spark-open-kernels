"""Freeze S11 and the self-contained CUPTI lab into one h200_lab payload."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
S11 = ROOT / 'problems/muse_decode_layer/cand_r3_best7_s11_final.py'
source = S11.read_text()
assert hashlib.sha256(source.encode()).hexdigest().startswith('acda6c62')
activity = (HERE / 'activity.cpp').read_text()
payload = ('import gc, hashlib, json, linecache, math, statistics, time, types, lzma, base64\n'
           'import torch\nfrom torch.utils.cpp_extension import load_inline\n'
           'TL_S11 = ' + repr(source) + '\nTL_ACTIVITY = ' + repr(activity) + '\n\n' +
           (HERE / 'selected_prepare.py').read_text() + '\n\n' +
           (HERE / 'lab_body.py').read_text())
compile(payload, 'lab_timeline.py', 'exec')
(HERE / 'lab_timeline.py').write_text(payload)
print('lab_timeline.py', hashlib.sha256(payload.encode()).hexdigest())
