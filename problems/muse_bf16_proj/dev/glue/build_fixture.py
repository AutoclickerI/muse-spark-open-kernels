"""Local fixture of the full owner-controlled best; all new paths forced for validation."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
base=(ROOT/'problems/muse_decode_layer/cand_r3_best.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2'
# The original file is neither edited nor submitted by this lane.
tail='''
_gb_fixture_prepare=prepare
_gb_fixture_original=kernel_fn
kernel_fn=gb_wrap_kernel(_gb_fixture_original,project,stage_attention,experts_v9)

def prepare(weights):
    started=time.monotonic();state=_gb_fixture_prepare(weights)
    callbacks=gb_callbacks(state,project,boundary,route_and_norm,o_and_pool)
    recipes={(m,p):[(min(m,8),64,2 if p=='b1' else 8,64 if p=='pool' else 128,4)]
             for m in GB_TOKENS for p in ('b1','pool','b2')}
    rows=gb_prepare(state,callbacks,started+240,recipes=recipes)
    bank=state['_gb_glue']
    # Test-only forced dispatch. This is never a submitted source or a speed recommendation.
    for key,choices in recipes.items():bank['plans'][key]=GbPlan(state,key[0],key[1],choices[0])
    bank['prefix']={m:True for m in GB_TOKENS};bank['enabled']={m:True for m in GB_TOKENS}
    for row in rows:print(row,flush=True)
    print('GB_FIXTURE force18 actual native fusion paths (same Triton code on sm120/sm90)',flush=True)
    return state
'''
path=ROOT/'runs/screen/verify_proj/glue_layer_fixture.py'
path.write_text(base+'\n'+(HERE/'projection_glue.py').read_text()+tail);print(path)
