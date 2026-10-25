"""Integrated S11 F4+LIW vs original S11 nf cell on the same fixtures."""
import hashlib
import importlib.util
import json
import runpy
from pathlib import Path
import torch

H = Path(__file__).resolve().parent
R = H.parents[4]


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def main():
    a=load('original',R/'problems/muse_decode_layer/cand_r3_best7_s11_final.py')
    b=load('wide',H/'s11_f4_wide.py')
    lab=runpy.run_path(str(R/'prof/lab/moe_nf_tune_m128.py'))
    rows=[]
    for seed in range(3):
        w=lab['dc_weights'](5353+7919*seed,'cuda')
        wp=dict(w,post_norm_w=w['post_expert_norm'])
        base=a.issue_prepare(a.nf_ilv(wp))
        native=a._prepare_recoded_bank(w,a.v9_prepare(wp),a._v9_recoded_module())
        banks=(base,b.nf_wide(dict(base)))
        for M in (128,256):
            ds=lab['dc_fixtures'](w,M);wants=lab['dc_oracles'](ds,w)
            for data,want in zip(ds,wants):
                outs=[]
                for mod,bank in zip((a,b),banks):
                    p=mod.nf_plan(data,bank,native,(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event()))
                    for fn in p['steps']:fn()
                    outs.append(p['out'].clone())
                torch.cuda.synchronize()
                errors=[lab['br_error'](o,want) for o in outs]
                assert all(e<=.015 and r<=.075 for e,r in errors),errors
                diff=float((outs[0].float()-outs[1].float()).abs().max())
                assert diff<=.01*float(outs[0].float().abs().max()),diff
                rows.append(dict(seed=seed,M=M,bit_equal=torch.equal(*outs),max_abs=diff,errors=errors))
        del banks,base,native,w,wp
        torch.cuda.empty_cache()
    result=dict(sha256=hashlib.sha256((H/'s11_f4_wide.py').read_bytes()).hexdigest(),rows=rows)
    (H/'wide_equiv.json').write_text(json.dumps(result,indent=2)+'\n')
    print('WIDE_EQUIV',len(rows),'bit',sum(r['bit_equal'] for r in rows),'max_abs',max(r['max_abs'] for r in rows))


if __name__=='__main__':main()
