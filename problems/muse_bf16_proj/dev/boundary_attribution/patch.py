"""S14 controls for boundary residence and MoE-to-unpool scheduling; no projection arithmetic changes."""
import ast
import hashlib
from pathlib import Path
import sys

BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def once(s,a,b):
    assert s.count(a)==1,(a[:80],s.count(a));return s.replace(a,b,1)


def patch(src,mode):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA
    assert mode in ('release','hint','release_hint','geometry','serial')
    tree=ast.parse(src);lines=src.splitlines(True);changes=[];suffix=''
    if mode in ('release','release_hint'):
        n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SMALLM_VAR_CUDA')
        code=ast.literal_eval(n.value)
        code=once(code,'  constexpr int RED = NT * 4;\n  const int npairs = M * 8;','''  constexpr int RED = NT * 4;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // The unpooler can issue immutable weight requests while this grid finishes.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  const int npairs = M * 8;''')
        changes.append((n.lineno-1,n.end_lineno,'SMALLM_VAR_CUDA = '+repr(code)+'\n'))
    if mode in ('hint','release_hint'):
        n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gb_kernel')
        kernel=ast.get_source_segment(src,n).replace('def _gb_kernel(','def _bd_unpool_kernel(',1)
        kernel=once(kernel,'    _pdl_wait(USE_PDL)',
            '    lines=tl.arange(0,BK//64)\n    _l2_prefetch(W+n[:,None]*K+kb0*BK+lines[None,:]*64)\n    _pdl_wait(USE_PDL)')
        suffix+='\n\n@triton.jit\n'+kernel+'\n'
        n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gb_gemv')
        code=ast.get_source_segment(src,n)
        code=once(code,'_launch(_gb_kernel,',
            '_launch(_bd_unpool_kernel if (T in (16,32) and N==8192 and K==4096) else _gb_kernel,')
        changes.append((n.lineno-1,n.end_lineno,code+'\n'))
    if mode=='serial':
        cu=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='GLUE_CL_CUDA'))
        cu=once(cu,'at[1].val.programmaticStreamSerializationAllowed = 1;','at[1].val.programmaticStreamSerializationAllowed = 0;')
        suffix+='\nBD_SERIAL_CUDA='+repr(cu)+'\n'+(HERE/'serial_runtime.inc').read_text()
    if mode in ('geometry','serial'):
        setup="plan['geometry']['b1']=('cl',4,256,1);plan['geometry']['b2']=('cl',4,256,1)" if mode=='geometry' else "plan['bd_serial']=True"
        warm='' if mode=='geometry' else '    bd_serial_module()\n'
        suffix+='''
_bd_base_prepare=prepare
def prepare(weights):
    state=_bd_base_prepare(weights)
'''+warm+'''    for key,plan in state['best7_plans'].items():
        if key!='other' and key[0] in (16,32) and key[1]==1:
            '''+setup+'''
    return state
'''
    for lo,hi,code in sorted(changes,reverse=True):lines[lo:hi]=[code]
    out=''.join(lines)+suffix;ast.parse(out);return out


if __name__=='__main__':
    src=(REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
    for mode in sys.argv[1:] or ('release','hint','release_hint','geometry','serial'):
        out=patch(src,mode);path=HERE/(mode+'_trial.py');path.write_text(out)
        print(mode,hashlib.sha256(out.encode()).hexdigest())
