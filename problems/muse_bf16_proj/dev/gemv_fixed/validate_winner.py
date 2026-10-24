"""Live-routing full-layer admission for receipt95ab's M1 O/config6 cell."""
import hashlib
import importlib.util
import json
import linecache
from pathlib import Path
import sys
import time

import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev'),str(ROOT/'problems/muse_decode_layer')]
import local_eval
import verify_best

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

p=load('gf_winner_lab',ROOT/'prof/lab/dense_gemv_fixed.py')
pp=load('gf_winner_setup',ROOT/'prof/lab/dense_pdl_prefetch.py')
cell_module=load('gf_winner_cell',HERE/'projection.py')
filename='<gf_winner_best7_scale>';source=p.GF_BASE_SOURCE
linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
ns={'__name__':'gf_winner_base','__builtins__':__builtins__};exec(compile(source,filename,'exec'),ns)
m=p.GFSpace(ns);forced=verify_best.install(m,False)
ref=local_eval.load_module('reference',ROOT/'problems/muse_decode_layer/reference.py')
weights=ref.generate_weights(ref.WEIGHTS_SEED);problem=local_eval.Problem(ref,weights)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
start=time.monotonic();state=pp.pp_state(m,weights)
state['moe_bank_raw']=state['moe_bank']
state['moe_bank']=m._prepare_recoded_bank(weights,state['moe_bank_raw'],m._v9_recoded_module())
cell=cell_module.Best7ProjectionCell(state['o'],1,6,fp32=False)
state['best7_stream_ops'][1,'o',6,1]=cell
rows=[]
for layer,window in (('full',0),('sliding',2048)):
    case=dict(batch=1,ctx=80000,layer=layer,q_len=1,spread='fixed',dist='soft',page=1)
    data=problem.make_input(case,172903)
    base=pp.pp_plan(m,(1,1,window));candidate=dict(base,projection={'o':('stream',6,1)})
    state['best7_plans']={(1,1,window):candidate,'other':base}
    gb,bo=p.gf_graph(lambda:m._best7_run(data,state,base,None))
    gc,co=p.gf_graph(lambda:m.kernel_fn(data,state))
    for seed in (172903,180822,188741):
        fresh=problem.make_input(case,seed)
        for key,value in fresh.items():
            if isinstance(value,torch.Tensor):data[key].copy_(value)
        gb.replay();gc.replay();torch.cuda.synchronize()
        errors=[p.gf_check(a,b,.003,'live full-layer cell') for a,b in zip(co[:2],bo[:2])]
        assert torch.equal(co[2],bo[2]),('live route IDs changed',layer,seed)
        ok,message=problem.check(data,co);assert ok,(layer,seed,message)
        rows.append(dict(layer=layer,seed=seed,hidden_relative=errors[0],stream_relative=errors[1],route_ids_exact=True,
                         official_gate=True,official_message=message))
        print('GF_WINNER_PASS',layer,seed,errors,message,flush=True)
    gb.reset();gc.reset()
record=dict(receipt='95ab383de42a4fae',baseline_sha256=hashlib.sha256(source.encode()).hexdigest(),
    projection=str((HERE/'projection.py').relative_to(ROOT)),projection_sha256=hashlib.sha256((HERE/'projection.py').read_bytes()).hexdigest(),
    role='o',tokens=1,config=6,fp32=False,checks=rows,forced_hopper_calls=forced,elapsed_s=time.monotonic()-start,
    h200_measured_key=[1,1,0],sliding='local agreement only; H200 whole-layer gain unmeasured')
(HERE/'winner_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print('GF_WINNER_ADMISSION',json.dumps(record),flush=True)
