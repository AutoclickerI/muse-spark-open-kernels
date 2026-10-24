"""Build a decode-layer adapter with the incumbent raw-scale V9 and a gated WGMMA producer."""
import ast
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prof/lab'))
from build_moe_real_routing import namespace


def main():
    base=(ROOT/'problems/muse_decode_layer/cand_r3_v9.py').read_text()
    raw=(HERE.parent/'v9_reuse/dropin.py').read_text()
    # Select the raw incumbent by namespace, independent of neighboring lane additions.
    nodes=[]
    for node in ast.parse(raw).body:
        if isinstance(node,ast.FunctionDef) and node.name.startswith('rv_'):nodes.append(node)
        elif isinstance(node,ast.Import) and any((a.asname or '').startswith('rv_') for a in node.names):nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id.startswith('rv_')
                                               for t in node.targets for n in ast.walk(t)):nodes.append(node)
    raw=ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'
    assert 'def rv_prepare(' in raw and 'def rv_kernel_fn(' in raw
    adapter='''
v9_prepare=rv_prepare
v9_kernel_fn=rv_kernel_fn
def experts_v9(x,wts,lids,state):
    return rv_kernel_fn({'hidden_states':x,'topk_ids':lids,'topk_weights':wts,
                         'config':{'local_count':32,'d_hidden':4096,'d_expert':4096,'local_start':0}},
                        state['moe_bank'])
'''
    start=base.index('# MoE V9: persistent resident expert kernel')
    stop=base.index('\nCUDA_SRC = r',start)
    base=base[:start]+raw+adapter+base[stop:]
    fixture=(ROOT/'problems/muse_decode_layer/moe_tune.py').read_text()
    fixture=fixture[:fixture.index('\ndef _graph_time')]
    oracle=(HERE.parent/'v9_reuse/reuse.py').read_text()
    oracle=oracle[oracle.index('def _reuse_fp32_impl'):]
    driver=(HERE/'lab_driver.py.inc').read_text()
    driver=driver[:driver.index('\ndef mw_phase_times')]
    parts=[base,(HERE/'engine.py').read_text(),namespace(fixture,'fx_'),oracle,
           driver,(HERE/'dropin_tail.py.inc').read_text()]
    src='\n\n'.join(parts)
    out=ROOT/'problems/muse_decode_layer/dev/parked/cand_r3_mixed_wgmma.py'
    out.write_text(src)
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest(),len(src))


if __name__=='__main__':main()
