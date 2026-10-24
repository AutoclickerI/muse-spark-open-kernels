"""Build native V9 seed portfolio; preserve the already accepted v1/v2 lab snapshots."""
import ast
import hashlib
from pathlib import Path
from build_moe_v9_reuse import ROOT,HERE,raw_v9,wide_v9,reference_generator
from build_moe_real_routing import namespace
from build_moe_hot_v9 import build as build_hot


def parts():
    build_hot()
    base=ROOT/'problems/muse_moe_w4a16/cand_v9.py'
    # Keep the queued V2 scale decoder as the matched control; only the new portfolio gets its fast decoder.
    old=(HERE/'moe_v9_reuse_v2.py').read_text();t=ast.parse(old)
    cuda=next(n.value.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='base_CUDA_SRC' for k in n.targets))
    native=wide_v9(raw_v9(base.read_text()))
    nodes=ast.parse(native)
    for n in nodes.body:
        if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='CUDA_SRC' for k in n.targets):n.value=ast.Constant(cuda)
    native=ast.unparse(nodes)+'\n'
    native=native.replace('nstage = 2\n','nstage = state.get("nstage_choice",2)\n')
    hot=(ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/hot_v9.py').read_text()
    oracle=(ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/reuse.py').read_text();oracle='import torch\n'+oracle[oracle.index('def _reuse_fp32_impl'):]
    fixture=(ROOT/'problems/muse_decode_layer/moe_tune.py').read_text().split('\ndef _graph_time')[0]
    return '\n'.join((namespace(native,'rv_'),namespace(hot,'hv_'),namespace(oracle,'rr_'),namespace(fixture,'fx_'),(HERE/'moe_seed_dropin.py.inc').read_text()))


def build():
    block='"""Native V9 MoE seed portfolio. FP32 prepare gate; byte-preserving NVFP4/E4M3 storage."""\n'+parts()
    out=ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/dropin.py';out.write_text(block)
    src=(ROOT/'problems/muse_decode_layer/cand_r3_v9.py').read_text();a=src.index('# MoE V9: persistent');b=src.index('\nCUDA_SRC = r"""',a);src=src[:a]+src[b:]
    src+='\n'+block+'''

def v9_prepare(weights):return moe_reuse_prepare(weights)


def experts_v9(x,wts,lids,state):return moe_reuse_experts(x,wts,lids,state['moe_bank'])
'''
    cand=ROOT/'problems/muse_decode_layer/cand_r3_v9_reuse.py';cand.write_text(src)
    for p in (out,cand):print(p.relative_to(ROOT),hashlib.sha256(p.read_bytes()).hexdigest())
    return block


if __name__=='__main__':build()
