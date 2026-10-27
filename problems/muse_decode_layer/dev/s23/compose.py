"""S23 = S22 phase 1 (fd5e0ca2) + P10's held-out projection ports and the P16 pool/router carrier, with the
d32S direct-input hybrid disabled (S22 phase-1 in-graph loss 0.99874 [0.99783,0.99965], 0e1f9bc8).

P10's patch (public_ports/patch.py, night/patch.py) builds its banks in prepare() wrappers. The paired lab
prepares every arm through the _s16_prepare_delta chain on a clone of the base state, so the same helper
bodies are appended verbatim and the same bank builds and warm calls run as the last delta step: after the
whole S22 prepare, as P10's outermost wrappers did. Until the banks exist the dispatch is S22's, as P10's
wrappers restored it during the base prepare.
"""
import hashlib,importlib.util,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1];PP=R/'problems/muse_bf16_proj/dev/public_ports'
def load(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s22=load(H.parent/'s22/compose.py','s23_s22_compose');s19=s22.s19;expand=s22.expand
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
BASE_SHA='fd5e0ca2ef4bbc1128b04b68aa6802c9543b61808e7663ca5ad10ec4d957110b'
BASE_SHIPPED='2287496067a3da750d43afd1943c12e69fb1d5337ce1c5e528d4b6c64649a6fd'
BASE_FILE='problems/muse_decode_layer/cand_r3_best7_s22p1_final.py'
# P10 P4_HANDOFF.json point names; P16 is night/body.py.inc's NG_GROUPS['P16'].
PORTS={'PORT2':(1,'unpooler_slice','P'),'PORT4':(16,'qkv_gate','N'),'PORT6':(16,'gate','P'),'PORT11':(256,'qkv_gate','N'),'PORT14':(256,'unpooler','N')}
NIGHT=('P16',)
GROUPS=(*PORTS,*NIGHT,'HYd32S')
ROWS={'HYd32S':['d32S'],'PORT2':['d1S','d1F'],'PORT4':['d16S','d16F'],'PORT6':['d16S','d16F'],'P16':['d16S','d16F'],'PORT11':['v32S','v32F'],'PORT14':['v32S','v32F']}
HELPERS=(PP/'port_body.py.inc',PP/'night/body.py.inc')
TAIL='''
S23_PORTS=PORTS_LITERAL
S23_NIGHT=NIGHT_LITERAL
_s23_base_project=_best7_project
_s23_base_gate=_gate_split_apply
def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    key=(x.shape[0],name)
    if 'night_gap' in state and any(key in NG_GROUPS[g] for g in S23_NIGHT):return ng_project(x,name,state['night_gap'])
    provider=S23_PORTS.get(key)
    if provider is not None and 'p10_public' in state:return pp_project(x,name,state['p10_public'],provider)
    return _s23_base_project(x,name,state,plan,out_dtype)
def _gate_split_apply(x,y,state):
    provider=S23_PORTS.get((x.shape[0],'gate'))
    if provider is not None and 'p10_public' in state:return pp_gate(x,y,state['p10_public'],provider)
    return _s23_base_gate(x,y,state)
'''
DELTA='''    bank=pp_prepare(globals(),weights,state,{(t,r,p) for (t,r),p in S23_PORTS.items()})
    for (t,r),p in S23_PORTS.items():
        w=bank['roles'][r];x=torch.zeros((t,w.shape[1]),device=w.device,dtype=torch.bfloat16)
        if r=='gate':pp_gate(x,torch.zeros((t,1024),device=w.device,dtype=torch.bfloat16),bank,p)
        else:pp_project(x,r,bank,p)
    state['p10_public']=bank
    if S23_NIGHT:
        night=ng_prepare(globals(),state)
        for t,r in sorted({k for g in S23_NIGHT for k in NG_GROUPS[g]}):
            w=night['roles'][r];ng_project(torch.zeros((t,w.shape[1]),device=w.device,dtype=torch.bfloat16),r,night)
        state['night_gap']=night
    torch.cuda.synchronize()'''

def base(hybrid_off=False):
 source,meta=s22.build(['v16F'],['d32S'] if hybrid_off else []);assert hybrid_off or meta['sha256']==BASE_SHA;return source

def selection():
 """P10's S22 selection, pinned: the points and groups this composition ships."""
 sel=json.loads((PP/'night/s22_selection.json').read_text())
 assert sorted(map(tuple,sel['points']))==sorted(PORTS.values()) and tuple(sel['groups'])==NIGHT and sel['pool']=={}
 return sel

def build(groups=GROUPS):
 groups=set(groups);assert groups<=set(GROUPS) and groups
 ports={(t,r):p for g,(t,r,p) in PORTS.items() if g in groups};night=tuple(g for g in NIGHT if g in groups)
 source=base('HYd32S' in groups)+'\n'+'\n'.join(p.read_text() for p in HELPERS)+TAIL.replace('PORTS_LITERAL',repr(ports)).replace('NIGHT_LITERAL',repr(night))
 source=s19.delta(source,'P10PORTS',DELTA);compile(source,'<S23>','exec')
 return source,dict(series='S23',sha256=sha(source),base_sha256=BASE_SHA,groups=sorted(groups),helpers={str(p.relative_to(R)):sha(p.read_text()) for p in HELPERS})

def arm_specs():
 out={'INC':dict(source=base(),rows=[],kind='incumbent')}
 for g in GROUPS:out['off_'+g]=dict(source=build(set(GROUPS)-{g})[0],rows=ROWS[g],kind='leave_out',group=g)
 return out

if __name__=='__main__':
 selection();source,meta=build();(H/'generated').mkdir(exist_ok=True);(H/'generated/s23.py').write_text(source);print(meta)
