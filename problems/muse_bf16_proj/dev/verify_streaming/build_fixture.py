"""Force the decode dispatch entry through portable TMA/mma.sync for local admission."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = HERE.parents[2]/'muse_decode_layer'/'cand_r3_best6.py'
out = HERE/'cand_verify_streaming_fixture.py'
out.write_text(base.read_text()+'\n\n'+(HERE/'projection.py').read_text()+'''
_vs_fixture_prepare = prepare
_vs_fixture_variant = _project_variant

def _project_variant(x,w,name,state,variant,out_dtype=torch.bfloat16):
    if isinstance(variant,tuple) and variant[0]=='verify_stream':
        return vs_project(x,w,name,state,out_dtype)
    return _vs_fixture_variant(x,w,name,state,variant,out_dtype)

def prepare(weights):
    state = _vs_fixture_prepare(weights)
    recipes = {}
    for m in (8,128,256):
        for name in ('qkv_gate','o','pool_router','unpooler_slice' if m<=8 else 'unpooler'):
            code = 2 if m==8 else (4 if m==128 else 6)
            choices = [s for c,s in vs_choices(m,*state[name].shape) if c==code]
            recipes[(m,name)] = (code,max(choices))
    vs_prepare_dispatch(state,recipes)
    for key,(code,split) in recipes.items():
        state['proj_plan'][key] = ('verify_stream',code,split)
    print('VERIFY_STREAM_FORCED',recipes,flush=True)
    return state
''')
print(out)
