"""S22 on accepted S21: restore d16S and fuse selected direct-input prologues."""
import ast,importlib.util,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
def load(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s21=load(H.parent/'s21/compose.py','s22_s21_compose');s19=s21.s19;sha=s21.sha;expand=s21.expand
BASE_SHA='109c0135b9a445b88b2efd1ebb73f15002a332ce553c8e50afb5557d6816398d'
BASE_SHIPPED='eacc087bd59b3cd58ee6b7690913868d86a8deb2ad1ff50040329ece62f7c8a4'
CELLS={'d32S':(32,1,True),'v1S':(1,8,True),'v1F':(1,8,False),'v16S':(16,8,True),'v16F':(16,8,False),'v32S':(32,8,True)}
HYBRIDS={'d32S','v1S'}
WARM="_s19_prior_warm=_s16_prepare_delta\ndef _s16_prepare_delta(weights,state):\n    state=_s19_prior_warm(weights,state)\n    import time as _t\n    _t0=_t.monotonic()\n    _s19_warm(state,weights['qkv_gate'].device)\n    state.setdefault('s19_prepare_s',{})['warm']=_t.monotonic()-_t0\n    return state\n"
def base():
 path=P/'cand_r3_best7_s21_final.py';assert sha(path.read_bytes())==BASE_SHIPPED
 source=expand(path);assert sha(source)==BASE_SHA;return source

def build(dropped=(),disabled_hybrids=()):
 dropped=set(dropped);disabled=set(disabled_hybrids);assert dropped<=CELLS.keys() and disabled<=HYBRIDS
 source=s19.swap(base(),WARM,'')
 picks=dict(s21.PUBLIC_SELECTION);picks.pop((16,1,True))
 for cell in dropped:picks.pop(CELLS[cell])
 source=s21.s20.assign(source,'_PUBLIC_SELECTION',picks)
 tail=(H/'inputs/hybrid.py.inc').read_text();assert sha(tail)==json.loads((H/'inputs/provenance.json').read_text())['tail_sha256']
 hy={CELLS[cell]:'F' for cell in sorted(HYBRIDS-dropped-disabled)}
 tail=s21.s20.assign(tail,'_HY_SELECTION',hy)
 old="def _hy_build():\n for name in sorted(_HY_SOURCES):_hy_module(name).setup(torch.device('cuda'))"
 new="""def _hy_build():
 names={'operator_direct' if key[2] and (arm=='O' or key[0]==1 and arm!='J') else 'lee_hybrid' for key,arm in _HY_SELECTION.items()}
 for name in sorted(names):_hy_module(name).setup(torch.device('cuda'))"""
 tail=s19.swap(tail,old,new);source+='\n'+tail
 source=s19.delta(source,'HYBRID','    _hy_build()')
 source=s19.delta(source,'warm',s19.WARM)
 compile(source,'<S22>','exec')
 return source,dict(series='S22',sha256=sha(source),base_sha256=BASE_SHA,dropped_cells=sorted(dropped),disabled_hybrids=sorted(disabled))

def arm_specs(dropped=(),disabled_hybrids=()):
 dropped=set(dropped);disabled=set(disabled_hybrids)
 out={'INC':dict(source=base(),rows=[],kind='incumbent')}
 for cell in CELLS.keys()-dropped:
  out['off_'+cell]=dict(source=build(dropped|{cell},disabled)[0],rows=[cell],kind='S19_cell',cell=cell)
 for cell in sorted(HYBRIDS-dropped-disabled):
  out['no_HY_'+cell]=dict(source=build(dropped,disabled|{cell})[0],rows=[cell],kind='prior_port',cell=cell)
 return dict(sorted(out.items(),key=lambda x:(x[0]!='INC',x[0])))

if __name__=='__main__':
 source,meta=build();(H/'generated').mkdir(exist_ok=True);(H/'generated/s22.py').write_text(source);(H/'selection.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
