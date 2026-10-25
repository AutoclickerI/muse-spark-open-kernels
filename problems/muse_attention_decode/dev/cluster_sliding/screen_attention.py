"""local_eval with the explicit N64 launch stand-in for the Hopper-only N128 persistent ring."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'tools'))
import local_eval
force='--forced' in sys.argv
if force:sys.argv.remove('--forced')
original=local_eval.load_module

def loader(name,path):
    m=original(name,path)
    if name!='submission':return m
    import torch
    if torch.cuda.get_device_capability()[0]!=9:
        m.p8_vq8_ext=m._vq8_ext
        print('LOCAL ONLY: frozen N64 module stands in for N128 persistent-ring launch; exact N128 math checked separately.',flush=True)
    if force:
        fn=m.kernel_fn
        def forced(data):
            c=data['config']
            if c['q_len']==8 and not c['window']:return m.p8_vq8_attention(data,stages=4,min_tiles=8)
            return fn(data)
        m.kernel_fn=forced
        if hasattr(m,'p5_head_probe'):
            m.P5_HEAD_SEEDS=m.P5_HEAD_SEEDS[:1];m.P5_HEAD_ROUNDS=2
            assert len(m.p5_head_probe())==2
    return m
local_eval.load_module=loader;local_eval.main()
