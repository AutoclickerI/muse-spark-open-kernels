"""Check the multi-draw regression and exact FA3 metadata forwarding on CPU."""
from pathlib import Path
import ast,json,sys,types
import torch

ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
tree=ast.parse((ROOT/'problems/muse_decode_layer/cand_r3_best6.py').read_text())
names={'_tune_attention_portfolio','_attention_fa3_public'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
seeds=(172903,180822,188741);layer=('layer',None);split=('fa3',(32,True))
ns={'torch':torch,'_ATTENTION_PUBLIC_SEEDS':seeds,'_attn_port_hopper':lambda device:True,
    '_best5_left':lambda *a:True,'_attn_port_ext':lambda:None,
    '_attn_port_xext':lambda kind:None,'_attn_port_p8ext':lambda:None,
    '_attn_port_candidates':lambda length:[layer,split],
    '_attn_port_synthetic':lambda *a:('gate',{}),
    '_attention_store_agrees':lambda *a:None,'_attention_reference':lambda *a:None,
    '_attention_agrees':lambda *a:(True,0.,0.),
    '_attn_port_variant':lambda qkv,data,state,choice:(choice,data.get('seed'))}
exec(compile(ast.Module(nodes,type_ignores=[]),'<candidate attention functions>','exec'),ns)
def draw(batch,length,window,device,seed,state):
    return None,{'seed':seed,'seq_lens':torch.arange(batch)+80000}
ns['_attention_public_draw']=draw

# An explicit split wins the first sample by 40%, then loses enough on the
# other public draws that its mean is slower. Test both q1 and q8 selection.
cost={seeds[0]:60.,seeds[1]:130.,seeds[2]:130.}
def measure(fn,flush,repeats):
    choice,seed=fn()
    return 100. if choice==layer else cost[seed]
ns['_measure_graph']=measure
state={};plan=ns['_tune_attention_portfolio'](state,None,'cpu')
assert len(plan)==12 and all(c==layer for c in plan.values()),plan
assert len(state['attention_public_rows'])==12
for row in state['attention_public_rows']:
    assert row['selected']==layer
    assert len(row['trials'])==2 and all(len(t['seed_us'])==3 for t in row['trials'])
    assert row['trials'][1]['seed_us']==[60.,130.,130.]

# Budget expiry after two optional samples must not promote a partial winner.
budget={'expired':False}
ns['_best5_left']=lambda *a:not budget['expired']
def partial(fn,flush,repeats):
    choice,seed=fn()
    if choice!=layer and seed==seeds[1]:budget['expired']=True
    return 100. if choice==layer else 1.
ns['_measure_graph']=partial
partial_state={};partial_plan=ns['_tune_attention_portfolio'](partial_state,None,'cpu')
assert all(c==layer for c in partial_plan.values())
assert len(partial_state['attention_public_rows'])==1
assert len(partial_state['attention_public_rows'][0]['trials'])==1

# Spy on the actual board-control body. Metadata is forwarded by identity,
# including None at q8; no explicit scheduler is constructed by this wrapper.
seen=[]
flash=types.ModuleType('sgl_kernel.flash_attn')
def call(**kw):seen.append(kw);return torch.zeros((8,16,64),dtype=torch.bfloat16)
flash.flash_attn_with_kvcache=call
sys.modules['sgl_kernel.flash_attn']=flash
ns.update(HKV=2,HQ=16,HD=64,Q_SIZE=1024,RMS_EPS=1e-6,
          _norm_gate_kernel=None,_launch=lambda *a,**kw:None)
for length,metadata in ((8,None),(8,object()),(1,object())):
    data={'config':dict(batch=1,q_len=length,tokens=length,window=0,kv_scale=1.,sm_scale=.125,qk_scale_factor=5.473),
          'scheduler_metadata':metadata,'k_cache':None,'v_cache':None,'page_table':None,
          'cache_seqlens':None,'cu_seqlens_q':None}
    ns['_attention_fa3_public'](torch.zeros((length,16,64)),None,data,{})
    assert seen[-1]['scheduler_metadata'] is metadata
result=dict(mean_over_draws_rejects_single_sample_winner=True,q1_and_q8_shapes=12,
            incomplete_trial_retains_control=True,board_metadata_identity_cases=3)
(HERE/'regression_local.json').write_text(json.dumps(result,indent=2)+'\n')
print('ATTENTION_REGRESSION',json.dumps(result))
