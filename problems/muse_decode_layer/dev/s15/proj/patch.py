"""Attributed NF projection handoff onto exact S14 final f1c77ce6. No baseline edits."""
import ast,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'


def patch(src,selected=()):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'requires exact S14 finalf1c77ce6'
    notice='\n'.join('# '+l for l in ('Nathen Fernandes; github.com/NathenFernandes/muse-bf16-proj @ 86aff75; MIT\n'+(H/'LICENSE.NathenFernandes').read_text()).splitlines())
    body=(H/'port.py').read_text().replace('NF_VENDOR = None','NF_VENDOR = '+repr((H/'nf_selected.py').read_text()))
    hook='''
# Selected by paired H200 producer+consumer / layer evidence. T1 coded rows remain ours.
NF_PORT_SELECTED = __SELECTED__
NF_PORT_FORCE = False
_nf_port_base_project = _project_variant
_nf_port_base_prepare = prepare
_nf_port_base_best7_project = _best7_project

def _project_variant(x,w,name,state,variant,out_dtype=torch.bfloat16):
    key=(x.shape[0],name)
    if 'nf_port' in state and (key in NF_PORT_SELECTED or NF_PORT_FORCE and key in NF_SMALL.keys()|NF_WIDE):
        return nf_project(x,name,state['nf_port'])
    return _nf_port_base_project(x,w,name,state,variant,out_dtype)

def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    key=(x.shape[0],name)
    if 'nf_port' in state and (key in NF_PORT_SELECTED or NF_PORT_FORCE and key in NF_SMALL.keys()|NF_WIDE):
        return nf_project(x,name,state['nf_port'])
    return _nf_port_base_best7_project(x,name,state,plan,out_dtype)

def prepare(weights):
    state=_nf_port_base_prepare(weights)
    state['nf_port']=nf_prepare(weights,None if NF_PORT_FORCE else NF_PORT_SELECTED)
    return state
'''.replace('__SELECTED__',('{'+', '.join(repr(tuple(x)) for x in sorted(selected))+'}') if selected else 'set()')
    out=src+'\n'+notice+'\n'+body+'\n'+hook
    assert len(out.encode())<1_000_000,(len(out.encode()),'1 MB limit')
    compile(out,'<S15 NF port>','exec')
    return out

if __name__=='__main__':
    base=H.parents[2]/'muse_decode_layer/cand_r3_best7_s14_final.py'
    selected=json.loads((H/'selection.json').read_text())['selected'] if (H/'selection.json').exists() else []
    out=Path(sys.argv[1]) if len(sys.argv)>1 else H/'cand_nf_s15.py'
    out.write_text(patch(base.read_text(),[tuple(x) for x in selected]))
    (H/'patch_manifest.json').write_text(json.dumps(dict(base_sha256=BASE_SHA,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,selected=selected),indent=2)+'\n')
    print(out,out.stat().st_size,hashlib.sha256(out.read_bytes()).hexdigest()[:8])
