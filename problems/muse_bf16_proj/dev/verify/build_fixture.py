"""Build a local-only full-layer integration fixture; never a queued candidate."""
from pathlib import Path
import ast
import pprint

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


def format_block(text):
    """Keep exact CUDA strings while making every added serving line <=400."""
    tree=ast.parse(text);lines=text.splitlines(keepends=True);edits=[]
    for n in tree.body:
        if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name):continue
        if max(len(s.rstrip('\n')) for s in lines[n.lineno-1:n.end_lineno])<=400:continue
        name=n.targets[0].id;value=ast.literal_eval(n.value)
        if isinstance(value,str):
            body='(\n'+''.join('    '+repr(value[i:i+96])+'\n' for i in range(0,len(value),96))+')'
        else:body=pprint.pformat(value,width=110,sort_dicts=False)
        edits.append((n.lineno-1,n.end_lineno,name+' = '+body+'\n'))
    for start,end,body in reversed(edits):lines[start:end]=[body]
    result=''.join(lines)
    assert ast.dump(ast.parse(result))==ast.dump(tree)
    assert max(map(len,result.splitlines()))<=400
    return result


def build():
    grouped='VG_SOURCE='+(HERE/'grouped_projection.py').read_text().split('VG_SOURCE=',1)[1]
    module=format_block((HERE/'verify_projection.py').read_text()+'\n'+grouped+'\n'+(HERE/'lab_library_tail.py').read_text().split('\ndef lab(')[0]+'\n'+(HERE/'h200_recipes.py').read_text()+'\n'+(HERE/'dropin_tail.py').read_text())
    (HERE/'projection_dropin.py').write_text(module)
    base=(ROOT/'problems/muse_decode_layer/cand_r3_v9_autotune_q1.py').read_text()
    # Repair two inherited static-check failures in this fixture only. The lane's
    # source stays frozen; the new block supplies neither this Lt entry nor FA3.
    base=base.replace('#include <cublasLt.h>','#include <ATen/cuda/CUDAContext.h>\n#include <cublasLt.h>')
    base=base.replace('lambda setting=setting: _fa3_call(q, gate, data, setting)',
                      'lambda setting=setting,q=q,gate=gate,data=data: _fa3_call(q, gate, data, setting)')
    tail='''
# Local integration fixture: force the native projection on target keys to
# verify whole-layer math; selection here is not a speed recommendation.
_vp_fixture_prepare = prepare
_vp_fixture_incumbent = project

def prepare(weights):
    state = _vp_fixture_prepare(weights)
    bank = {'incumbent':_vp_fixture_incumbent, 'plan':{}, 'layouts':{}, 'tma':vp_tma_module(),
            'grouped':vg_tma_module(), 'lt':_pg_native(),
            'workspace':torch.empty(4<<30,device=state['o'].device,dtype=torch.uint8)}
    for m in (128,256):
        for name in ('qkv_gate','pool_router','unpooler','o'):
            w=state[name];dtype=torch.float32 if name=='pool_router' else torch.bfloat16
            if name=='qkv_gate':recipe=('grouped',(128,1,1,2,1,2,1,1,2))
            elif name=='pool_router':recipe=('grouped',(32,1,1,2,2,4,1,1,4))
            elif name=='unpooler':recipe=('tma',(128,1,1,4,1,2,1,1))
            else:
                codes,_=bank['lt'].options(m,w.shape[0],w.shape[1],1,False,bank['workspace'].numel())
                bank['layouts'][(m,name,1)]=(w.T.contiguous(),codes)
                recipe=('lt',1,0)
            x=torch.randn(m,w.shape[1],device=w.device,dtype=torch.bfloat16)
            bank['name']=name;_vp_recipe(x,w,recipe,dtype,bank)
            bank['plan'][(m,name)]=(recipe,w.data_ptr(),tuple(w.shape),tuple(w.stride()),dtype)
    state['_verify_proj']=bank
    return state

project = verify_proj_bind(_vp_fixture_incumbent)
'''
    path=ROOT/'runs/screen/verify_proj/integration_fixture.py'
    path.write_text(base+'\n'+module+tail)
    return path


if __name__=='__main__':print(build())
