"""Local-only full decode fixture; strict portable stand-in for Hopper launches.

Not a candidate and never sent. No claim that Hopper SASS ran on sm_120.
"""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
base=(ROOT/'problems/muse_decode_layer/cand_r3_v9_autotune_q1.py').read_text()
# Same fixture-only inherited repairs as the previously admitted Lt integration.
base=base.replace('#include <cublasLt.h>','#include <ATen/cuda/CUDAContext.h>\n#include <cublasLt.h>')
base=base.replace('lambda setting=setting: _fa3_call(q, gate, data, setting)',
                  'lambda setting=setting,q=q,gate=gate,data=data: _fa3_call(q, gate, data, setting)')
src=(HERE/'validate.py').read_text();tree=ast.parse(src);lines=src.splitlines(keepends=True)
def copy_class(name):
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    return ''.join(lines[node.lineno-1:node.end_lineno])
standin='\n'+copy_class('AbiStandin')+'\nabi=AbiStandin()\n'
standin+=copy_class('Plan').replace('lab.VgcPlan','VgcPlan')+'\n'+copy_class('Module')+'\n'
standin=standin.replace('assert sk>=1 and (sk==1 or mode in (0,2)) and sms<=132',
                        'assert sk>=1 and (sk==1 or mode in (0,2)) and sms>0')
standin+='''
_vgc_fixture_module=Module()
vgc_module=lambda:_vgc_fixture_module
_vgc_fixture_prepare=prepare
_vgc_fixture_project=project
project=vgc_bind(_vgc_fixture_project)

def prepare(weights):
    started=time.monotonic()
    state=_vgc_fixture_prepare(weights)
    recipes={(m,name):[(tile,mode,sk,132,0,1,0)]
             for m in (128,256)
             for name,tile,mode,sk in (('qkv_gate',0,3,1),('pool_router',1,2,4),
                                      ('unpooler',6,3,1),('o',7,2,2))}
    # Exercise the real optional selection with real cold graph timings. The
    # stand-in is torch math, so it need not beat the incumbent on sm_120.
    rows=vgc_prepare(state,_vgc_fixture_project,started+240,recipes)
    bank=state['_vgc_proj']
    for key,saved in list(bank['plans'].items()):
        saved[0].close();bank['plans'].pop(key)
    # Force all target outputs for the whole-layer correctness/graph tests.
    for (m,name),choices in recipes.items():
        w=state[name];dtype=torch.float32 if name=='pool_router' else torch.bfloat16
        plan=_vgc_fixture_module.plan(m,w.shape[0],w.shape[1],dtype,choices[0],w.device)
        bank['plans'][m,name]=(plan,w.data_ptr(),tuple(w.shape),tuple(w.stride()),dtype)
    for row in rows:print(row,flush=True)
    print('VGC_PORTABLE_FIXTURE forced8; Hopper transport is a stand-in',flush=True)
    return state
'''
# Plan.run in the test uses super() on the actual runtime class, not a fake ABI.
path=ROOT/'runs/screen/verify_proj/cutlass_layer_fixture.py'
path.write_text(base+'\n'+(HERE/'projection_cutlass.py').read_text()+standin)
print(path)
