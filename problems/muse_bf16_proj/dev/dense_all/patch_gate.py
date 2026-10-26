"""Apply only P10's held-out G0 winner: exact12 gate at T1 sliding.

    python3 patch_gate.py BASE.py OUTPUT.py
Native H200 e364381789e8445e, frozen S19 R1.00440465 [1.00300344,1.00580781].
P4 must confirm its composition; no merge or other coded rows enter this patch.
"""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));import patch

def apply(source):
    result=patch.patch(source,{(1,'gate_apply'):3})
    old="if key in DA_SELECTED and 'da' in state:return da_project(x,'gate'"
    assert result.count(old)==1
    result=result.replace(old,"if state.get('da_window',False) and key in DA_SELECTED and 'da' in state:return da_project(x,'gate'",1)
    result+='''
_da_gate_window_run=globals()["_best7_run"]
def _best7_run(data,state,plan,prefetch=None):
    state['da_window']=bool(data['config']['window'])
    return _da_gate_window_run(data,state,plan,prefetch)
'''
    compile(result,'<P10 selected gate>','exec');return result
if __name__=='__main__':
    assert len(sys.argv)==3
    source=Path(sys.argv[1]).read_text();result=apply(source);Path(sys.argv[2]).write_text(result)
    print(json.dumps(dict(base_sha256=hashlib.sha256(source.encode()).hexdigest(),source_sha256=hashlib.sha256(result.encode()).hexdigest())))
