"""Exact h200_lab wrapper through local_eval; forced native dispatch uses explicit resident math stand-ins."""
import ast,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
import local_eval,local_standin
forced='--forced' in sys.argv
if forced:sys.argv.remove('--forced')
original=local_eval.load_module

def load(name,path):
    m=original(name,path)
    if name!='submission':return m
    import torch
    m._lab_ran=True  # import-time lab is separately executed by check_lab.py
    if forced:
        m.VR_LOCAL_STANDIN=local_standin.install(m.__dict__,'vr_')
        base=m._VR_BASE_FN
        def kernel_fn(data):
            cfg=data['config']
            if cfg['q_len']==8 and cfg['window'] and cfg['num_q_heads']==8*cfg['num_kv_heads']:
                return m.vr_attention(data,name='C4P' if cfg['batch']==32 else 'C8P')
            return base(data)
        m.kernel_fn=kernel_fn
    return m
local_eval.load_module=load
sys.argv=[sys.argv[0],str(H/'out/board_local.py'),'--no-baseline','--reps','2','--inputs','1','--json',str(H/'out'/('forced_board.json' if forced else 'normal_board.json'))]
local_eval.main()
