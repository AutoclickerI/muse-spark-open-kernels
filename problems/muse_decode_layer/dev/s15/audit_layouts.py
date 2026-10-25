"""Materialize every selected NF weight layout, including Hopper-only local launch rows.

Original packers assert bit-exact BF16 inverse reconstruction before returning.
The audit checks final payload dtypes, independently cloned codes and raw BF16 O32,
and records metadata. These temporary test banks are never added to prepare state.
"""
import hashlib
import json
import sys
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parents[1])]
import local_eval


def main():
    m=local_eval.load_module('submission',H.parents[1]/'cand_r3_best7_s15.py')
    ref=local_eval.load_module('reference',H.parents[1]/'reference.py')
    weights=ref.generate_weights(ref.WEIGHTS_SEED)
    torch=m.torch
    roles={'qkv_gate':weights['qkv_gate'],'qkv3':weights['qkv_gate'][:1280].contiguous(),
           'o':weights['o'],'unpooler':weights['unpooler']}
    ns=m._nf_vendor();rows={}
    for t,name in sorted(m.NF_PORT_SELECTED):
        weight=roles[name].clone()
        assert torch.equal(weight.view(torch.int16),roles[name].view(torch.int16))
        if (t,name) in m.NF_SMALL:
            family,cfg,*_=m.NF_SMALL[t,name]
            pk,tables,stream=m._nf_group_pack(ns,weight,name,family)
            assert pk is not None
            sp=stream(pk,1,*cfg)
            assert sp is not None,(t,name)
            payload=sp['wp'].clone();esc=sp['esc'].clone()
            assert payload.dtype==torch.uint8 and esc.dtype==torch.int32
            row=dict(family=family,code_bytes=payload.numel(),escape_bytes=esc.numel()*4,
                     table_dtype=str(tables.dtype),packer_inverse='all BF16 bits checked inside exact selected packer')
        elif (t,name)==(32,'unpooler'):
            pk=ns['dd_decodediet_tt_mw_pack_weight'](weight,64,4,1,True)
            assert pk is not None and pk['pack_exact'] and pk['compressed']
            assert pk['w'].dtype==torch.uint8 and pk['esc'].dtype==torch.int32
            row=dict(family='mw',code_bytes=pk['w'].numel(),escape_bytes=pk['esc'].numel()*4,
                     packer_inverse='all BF16 bits checked inside exact selected packer')
        elif (t,name)==(32,'o'):
            packed=ns['b2o_pack_olayer'](weight)
            assert packed.dtype==torch.bfloat16
            # Invert the row-dependent XOR swizzle and transpose, independently
            # checking every BF16 bit at its original coordinate.
            col=torch.arange(64,device=weight.device)[None,:]^((torch.arange(64,device=weight.device)[:,None]&7)*8)
            restored=packed.gather(-1,col.expand(128,16,64,64)).permute(0,2,1,3).contiguous().reshape_as(weight)
            assert torch.equal(restored.view(torch.int16),weight.view(torch.int16))
            row=dict(family='ol',dtype=str(packed.dtype),bytes=packed.numel()*2,exact_BF16_inverse=True)
        else:raise AssertionError((t,name))
        rows[str((t,name))]=row
        print('LAYOUT_PASS',t,name,row,flush=True)
    assert len(rows)==9
    out=dict(candidate_sha256=hashlib.sha256((H.parents[1]/'cand_r3_best7_s15.py').read_bytes()).hexdigest(),
             projection_patch_sha256=hashlib.sha256((H/'proj/patch.py').read_bytes()).hexdigest(),
             passed=True,rows=rows,local_only=True)
    (H/'hopper_layout_inventory.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
