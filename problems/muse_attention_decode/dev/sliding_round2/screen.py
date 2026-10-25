"""Official local_eval screen of the exact board packet; optional forced dispatcher covers every arm."""
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
import local_eval
original=local_eval.load_module
forced='--forced' in sys.argv
if forced:sys.argv.remove('--forced')
def load(name,path):
    m=original(name,path)
    if name=='submission' and forced:
        original_call=m.kernel_fn
        # Force every new shape arm even on a short TASK case. Actual native
        # WGMMA is compiled forsm90a; sm120 runs its exact mma.sync math shim.
        def kernel_fn(data):
            cfg=data['config'];key=cfg['batch'],cfg['q_len']
            if cfg['window'] and cfg['q_len'] in (1,8) and cfg['num_q_heads']==8*cfg['num_kv_heads']:
                arm=m.r2_SELECTED[(16,1) if cfg['q_len']==1 else (1,8)]
                return m.r2_apply_arm(data,None,arm)
            return original_call(data)
        m.kernel_fn=kernel_fn
    return m
local_eval.load_module=load
sys.argv=[sys.argv[0],str(H/'out/sliding_round2_board.py'),'--no-baseline','--reps','2','--inputs','1','--json',str(H/'out'/('forced_board.json' if forced else 'normal_board.json'))]
local_eval.main()
