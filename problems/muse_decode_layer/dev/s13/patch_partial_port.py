"""Apply the direct projection-partials attention port to frozen final S11.
Decode-1 sliding and full won the final-S11 whole-layer pair c114c35e787f46d1 (16 seeds).
Default rows are exactly those two; other rows are experimental when explicitly passed.
"""
import hashlib
import sys
from pathlib import Path
import partial_port

HERE=Path(__file__).resolve().parent
BASE_SHA='240248b2c27c890bf5055a77e87d2fc9c4429b747a04d714e51d5f67e936bb3e'
DEFAULT_ROWS=((1,1,False),(1,1,True))


def patch(src, rows=DEFAULT_ROWS, fast=False):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA, 'apply onto frozen cand_r3_best7_s12_final.py'
    old="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n"
    new=old+"    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in state['partial_port_rows']:\n        return _layer_partial_port(qkv,data)\n"
    assert src.count(old)==1
    src=src.replace(old,new)
    old='        native_pool.submit(_vq8_module)  # vq8 rows (q_len 8)\n'
    assert src.count(old)==1
    src=src.replace(old,old+'        native_pool.submit(partial_port_ext,_PP_FAST)\n')
    old="    state['vq8_rows']=_vq8_prepare(device)\n"
    assert src.count(old)==1
    src=src.replace(old,old+"    state['partial_port_rows']=_partial_port_prepare(device)\n")
    cell=partial_port.cell()
    # Include only the selected kernel in the production candidate.
    chosen=partial_port.source(fast)
    a=cell.index('PARTIAL_CUDA = ');b=cell.index('PARTIAL_CPP = ')
    cell=cell[:a]+'PARTIAL_CUDA = '+repr(chosen)+'\nPARTIAL_FAST_CUDA = PARTIAL_CUDA\n'+cell[b:]
    return src+'\n\n'+cell+f'\n_PP_ROWS = {rows!r}\n_PP_FAST = {fast!r}\n'+r'''

def _layer_partial_port(qkv,data):
    cfg=dict(data['config'],num_q_heads=HQ,num_kv_heads=HKV)
    return attention_partial_port(qkv,dict(data,config=cfg),_PP_FAST)


def _partial_port_prepare(device):
    for batch,length,sliding in _PP_ROWS:
        for ctx in (257,2305):
            qkv,data=_attn_port_synthetic(batch,length,2048 if sliding else 0,device,ctx)
            got=_layer_partial_port(qkv,data)
            _attention_store_agrees(qkv,data)
            ok,error,bound=_attention_agrees(got,_attention_reference(qkv,data),_attention_reference(qkv,data,True))
            if not ok:raise RuntimeError(f'partial_port {(batch,length,sliding)} ctx {ctx}: {error:.5f}>{bound:.5f}')
    print('PARTIAL_PORT_ROWS',_PP_ROWS,flush=True)
    return _PP_ROWS
'''

if __name__=='__main__':
    src=Path(sys.argv[1]).read_text()
    out=patch(src,fast='--fast' in sys.argv)
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2],hashlib.sha256(out.encode()).hexdigest())
