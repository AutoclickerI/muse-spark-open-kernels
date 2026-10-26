"""Append only selected P10 QKV kernels to P4's current layer source.

The patch preserves the existing gate stream, attention, QKV norms and all
neighbour fusion. Compilation/repacking run once in prepare, not timed calls.
"""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
SELECT={(8,2304):4,(16,2304):4,(16,1280):4}
RUNTIME='''
# P10 QKV exact12 dispatch; H200 2b45988c88fb42a5, stage selection only.
SGQ_SELECTED={(8,2304):4,(16,2304):4,(16,1280):4}
SGQ_LOCAL_REDUCE=False
_sgq_layer_prepare=prepare
_sgq_layer_project=_project_variant

def prepare(weights):
    st=_sgq_layer_prepare(weights)
    st['sgq']=sgq_prepare(weights,SGQ_LOCAL_REDUCE,SGQ_SELECTED)
    return st

def _project_variant(x,w,name,state,variant,out_dtype=torch.bfloat16):
    key=(x.shape[0],w.shape[0])
    if name in ('qkv_gate','qkv3') and key in SGQ_SELECTED and 'sgq' in state:
        return sgq_project(x,key[1],SGQ_SELECTED[key],state['sgq'])
    return _sgq_layer_project(x,w,name,state,variant,out_dtype)
'''
def patch(source):
    assert 'def kernel_fn(' in source
    assert 'SGQ_SELECTED=' not in source,'P10 QKV patch already installed'
    # P4 ships compressed source. All original kernels execute in the module's
    # globals; explicit aliases also keep its static Python check meaningful.
    aliases='\nimport torch,hashlib\n_nf_vendor=globals()["_nf_vendor"]\n_nf_group_pack=globals()["_nf_group_pack"]\n_project_variant=globals()["_project_variant"]\n'
    helper=(H/'qkv_dispatch.py.inc').read_text()
    # Retain the exact experimental recipe and its source pin; the integration
    # build only prunes unused template instances/banks from untimed prepare.
    changes={
        'def sgq_source(local_reduce=False):':'def sgq_source(local_reduce=False,selection=None):',
        '            for arm,(wk,s,stages) in enumerate(SGQ_CONFIGS):\n':'            for arm,(wk,s,stages) in enumerate(SGQ_CONFIGS):\n                if selection is not None and selection.get((t,n))!=arm:continue\n',
        'def sgq_prepare(weights,local_reduce=False):':'def sgq_prepare(weights,local_reduce=False,selection=None):',
        'ns,insts,plans=sgq_source(local_reduce);source=':'ns,insts,plans=sgq_source(local_reduce,selection);source='
    }
    for old,new in changes.items():
        assert helper.count(old)==1,old
        helper=helper.replace(old,new)
    out=source+aliases+helper+'\n'+RUNTIME
    compile(out,'<P10 QKV layer patch>','exec')
    return out
if __name__=='__main__':
    assert len(sys.argv)==3,'patch_qkv.py BASE.py OUTPUT.py'
    src=Path(sys.argv[1]).read_text();out=patch(src);Path(sys.argv[2]).write_text(out)
    print(json.dumps(dict(base_sha256=hashlib.sha256(src.encode()).hexdigest(),source_sha256=hashlib.sha256(out.encode()).hexdigest(),bytes=len(out.encode()),selection=sorted((t,n,a) for (t,n),a in SELECT.items()))))
