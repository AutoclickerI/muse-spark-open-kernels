"""Exact-S14 boundary rebuilds; publication changes only the decode unpooler/b2 edge."""
import ast,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
KINDS=('moment','exact','serial','route','chain','publish')

def configure(plan,kind,decode):
    plan.update(br_roles=(),br_mode=2,br_nt=512,br_route_nt=256,br_serial=0,br_pub=False)
    if kind=='publish':
        if decode:plan.update(br_roles=('b2',),br_pub=True)
    elif kind=='route':plan['br_roles']=('route',)
    else:
        plan['br_roles']=('b1','b2','route') if kind=='chain' else ('b1','b2')
        plan['br_mode']=1 if kind=='moment' else 2
        plan['br_serial']=int(kind=='serial')

def patch(base,kind):
    assert hashlib.sha256(base.encode()).hexdigest()==BASE_SHA and kind in KINDS
    suffix='\nBR_CUDA='+repr((HERE/'kernels.cu.inc').read_text())+'\n'+(HERE/'runtime.inc').read_text()
    if kind=='publish':
        node=next(n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef) and n.name=='_gb_kernel')
        kernel=ast.get_source_segment(base,node).replace('def _gb_kernel(X, W, P,','def _br_publish_kernel(X, W, P, FLAGS,',1)
        kernel=kernel.replace('    _pdl_start(USE_PDL)', '''    flag=FLAGS+ps*(N//BN)+pn
    tl.inline_asm_elementwise("st.release.gpu.global.u32 [$1],0; mov.u32 $0,0;", "=r,l", [flag], dtype=tl.int32, is_pure=False, pack=1)
    tl.debug_barrier()
    _pdl_start(USE_PDL)''',1)
        # All producer threads fence their own stores before publishing the flag.
        kernel+='''
    tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0,0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
    tl.debug_barrier()
    tl.inline_asm_elementwise("st.release.gpu.global.u32 [$1],1; mov.u32 $0,0;", "=r,l", [flag], dtype=tl.int32, is_pure=False, pack=1)
'''
        suffix+='\n@triton.jit\n'+kernel+'\n'+(HERE/'publish_runtime.inc').read_text()
    suffix+='\n'+__import__('inspect').getsource(configure)+'\n'
    suffix+='''
_br_prepare_base=prepare
def prepare(weights):
    state=_br_prepare_base(weights);br_module()
    for key,plan in state['best7_plans'].items():
        if key!='other' and key[0] in (16,32) and key[1] in (1,8):configure(plan,BR_KIND,key[1]==1)
    return state
'''
    out=base+'\n'+suffix+'\nBR_KIND='+repr(kind)+'\n';ast.parse(out);return out

if __name__=='__main__':
    base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
    for kind in KINDS:
        out=patch(base,kind);(HERE/(kind+'_trial.py')).write_text(out);print(kind,hashlib.sha256(out.encode()).hexdigest())
