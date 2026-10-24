"""Build a local-only attention harness with full q8 selecting the native prototype."""
from pathlib import Path
from build import build
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
source,cpp,driver=build()
base=(ROOT/'problems/muse_attention_decode/cand_verify8_flashdecode.py').read_text()
base=base.replace('def kernel_fn(data: Inputs)','def incumbent_kernel_fn(data: Inputs)')
# Portable screens keep the unchanged control outside full q8. This file is
# local admission only and is not sent before a same-shape H200 FA3 win.
base+='\nNATIVE_CUDA='+repr(source)+'\nNATIVE_CPP='+repr(cpp)+'\nimport hashlib\n'+driver
base+='''

def kernel_fn(data: Inputs):
    cfg=data['config']
    if cfg['q_len']==8 and not cfg['window']:
        return native_attention(data['qkv'],data,force_hopper=True)[0]
    return incumbent_kernel_fn(data)
'''
(HERE/'local_screen.py').write_text(base)
print(HERE/'local_screen.py')
