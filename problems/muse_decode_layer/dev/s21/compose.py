"""S21: per-cell public attention winners plus C3, gate, v32F combine and native layouts."""
import ast,importlib.util,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s20=load(H.parent/'s20/compose.py','s21_s20_compose')
s19=s20.s19;sha=s20.sha;BASE_SHA=s20.BASE_SHA
DEFAULT=('C3','GATE','VQ32','NP','PUBLIC')
ROWS={'C3':s20.ROWS['C3'],'GATE':s20.ROWS['GATE'],'VQ32':['v32F'],'NP':['v16S','v16F'],
      'PUBLIC':['d16S','d32S','v1S','v1F','v16S','v16F','v32S']}
PUBLIC_SELECTION={(16,1,True):'lee',(32,1,True):'lee',(1,8,True):'operator',
                  (1,8,False):'lee',(16,8,True):'lee',(16,8,False):'lee',(32,8,True):'lee'}
base=s20.base;expand=s20.expand

def without_prepare(tail,aliases):
    lines=tail.splitlines(True)
    for node in reversed(ast.parse(tail).body):
        if isinstance(node,ast.FunctionDef) and node.name=='prepare':lines[node.lineno-1:node.end_lineno]=[]
    tail=''.join(lines)
    for name in aliases:tail=s19.swap(tail,name+'=prepare\n','')
    return tail

def chain_tail(enabled):
    original=(H/'inputs/chain.py.inc').read_text()
    tail=without_prepare(original,('_p5b_sched_original_prepare','_p5b_merge_original_prepare','_p5b_vq_original_prepare'))
    picks={(16,8,True):4,(1,8,False):64} if 'FA_CHAIN' in enabled else {}
    fa={(16,8,True):'R4w4',(1,8,False):'R64'} if 'FA_CHAIN' in enabled else {}
    vq={**({(16,8,False):'Rw4'} if 'VQ16' in enabled else {}),**({(32,8,False):'Rw4'} if 'VQ32' in enabled else {})}
    for name,value in (('_P5B_FA_PICKS',picks),('_P5B_MERGE_SELECTION',fa),('_P5B_VQ_SELECTION',vq)):
        # _P5B_FA_PICKS shares a semicolon-delimited assignment line.
        if name=='_P5B_FA_PICKS':
            tail=s19.swap(tail,"_P5B_FA_PICKS={(16, 8, True): 4, (1, 8, False): 64}",name+'='+repr(value))
        else:tail=s20.assign(tail,name,value)
    # The remaining v32F VQ combine needs only the row reducer. Build the FA3
    # bridge and metadata only in the public-port leave-out, where FA3 is used.
    tree=ast.parse(tail);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_p5b_merge_setup')
    lines=tail.splitlines(True);body=lines[node.lineno-1:node.end_lineno]
    first=next(i for i,s in enumerate(body) if 'import sgl_kernel.flash_ops' in s)
    last=next(i for i,s in enumerate(body) if '_P5B_BRIDGE.fa_bind' in s)
    body[first:last+1]=['    if _P5B_MERGE_SELECTION:\n']+['    '+s for s in body[first:last+1]]
    lines[node.lineno-1:node.end_lineno]=body;tail=''.join(lines)
    return tail

def build(enabled=DEFAULT):
    enabled=set(enabled);assert enabled<=set(DEFAULT)
    effective=set(enabled)
    # Incremental public-port leave-out restores the pre-port S21 choices at
    # v16S/v1F/v16F; the primary comparison remains exact S19 at every cell.
    if 'PUBLIC' not in enabled:effective.update(('FA_CHAIN','VQ16'))
    core=[p for p in ('C3','GATE') if p in enabled]
    source,meta=s20.build(core)
    warm="_s19_prior_warm=_s16_prepare_delta\ndef _s16_prepare_delta(weights,state):\n    state=_s19_prior_warm(weights,state)\n    import time as _t\n    _t0=_t.monotonic()\n    _s19_warm(state,weights['qkv_gate'].device)\n    state.setdefault('s19_prepare_s',{})['warm']=_t.monotonic()-_t0\n    return state\n"
    source=s19.swap(source,warm,'')
    if effective & {'FA_CHAIN','VQ16','VQ32'}:
        source+='\n'+chain_tail(effective)+'''
def _s21_chain_build():
    _p5b_merge_setup()
    for role in _P5B_VQ_SOURCES:_p5b_vq_module(role)
_S19_BUILDS.append(_s21_chain_build)
'''
        source=s19.delta(source,'CHAIN',"    if _P5B_REDUCE is None or (_P5B_MERGE_SELECTION and _P5B_BRIDGE is None):_p5b_merge_setup()\n    for role in _P5B_VQ_SOURCES:_p5b_vq_module(role)\n    if _P5B_FA_PICKS:_p5b_sched_init(weights['qkv_gate'].device)")
    if 'NP' in enabled:
        source+='\n'+without_prepare((H/'inputs/native_layout.py.inc').read_text(),('_np_layout_prepare',))
        source=s19.delta(source,'NP',"    state['np']=np_prepare(state)")
    if 'PUBLIC' in enabled:
        tail=without_prepare((H/'inputs/public_ports.py.inc').read_text(),('_public_original_prepare',))
        source+='\n'+s20.assign(tail,'_PUBLIC_SELECTION',PUBLIC_SELECTION)+'''
def _s21_public_build():
    for name in sorted(set(_PUBLIC_SELECTION.values())):_public_module(name).setup(torch.device('cuda'))
_S19_BUILDS.append(_s21_public_build)
'''
        source=s19.delta(source,'PUBLIC',"    _s21_public_build()")
    source=s19.delta(source,'warm',s19.WARM)
    meta.update(enabled=[p for p in DEFAULT if p in enabled],series='S21')
    meta['sha256']=sha(source);compile(source,'<S21>','exec')
    return source,meta

if __name__=='__main__':
    (H/'generated').mkdir(exist_ok=True)
    s,m=build();(H/'generated/s21.py').write_text(s);(H/'selection.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
