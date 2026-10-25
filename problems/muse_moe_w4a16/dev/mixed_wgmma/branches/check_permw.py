"""PERMW integration against S12's own F4+wide cell and b88d04fe's arm."""
import hashlib
import importlib.util
import json
import runpy
from pathlib import Path
import torch

H=Path(__file__).resolve().parent
R=H.parents[4]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    a=load('s12',R/'problems/muse_decode_layer/cand_r3_best7_s12_final.py')
    b=load('s12perm',H/'s12_permw.py')
    lab=runpy.run_path(str(R/'prof/lab/moe_p3b_adapt_m128.py'))
    rows=[]
    for seed in range(3):
        w=lab['dc_weights'](5353+7919*seed,'cuda');wp=dict(w,post_norm_w=w['post_expert_norm'])
        bank=lab['br_prepare'](w)
        base=a.issue_prepare(a.nf_ilv(wp))
        perm=b.nf_perm(b.nf_wide(dict(base)))
        for M in (128,256):
            ds=lab['dc_fixtures'](w,M);wants=lab['dc_oracles'](ds,w)
            for data,want in zip(ds,wants):
                outs=[]
                for mod,bb in ((a,base),(b,perm)):
                    p=mod.nf_plan(data,bb,bank['native'],(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event()))
                    for f in p['steps']:f()
                    outs.append(p['out'].clone())
                outs.append(lab['br_call'](data,bank,'PERMW').clone())
                torch.cuda.synchronize()
                errors=[lab['br_error'](o,want) for o in outs]
                assert all(e<=.015 and r<=.075 for e,r in errors),errors
                diff=max(float((o.float()-outs[1].float()).abs().max()) for o in (outs[0],outs[2]))
                assert diff<=.01*float(outs[1].float().abs().max()),diff
                rows.append(dict(seed=seed,M=M,bit_equal_s12=torch.equal(outs[0],outs[1]),
                                 bit_equal_measured_arm=torch.equal(outs[1],outs[2]),max_abs=diff,errors=errors))
        del bank,base,perm,w,wp
        torch.cuda.empty_cache()
    result=dict(candidate_sha256=hashlib.sha256((H/'s12_permw.py').read_bytes()).hexdigest(),
                patch_sha256=hashlib.sha256((H/'patch_s12_permw.py').read_bytes()).hexdigest(),rows=rows)
    (H/'permw_equiv.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PERMW_EQUIV',len(rows),'S12 bit',sum(r['bit_equal_s12'] for r in rows),
          'measured arm bit',sum(r['bit_equal_measured_arm'] for r in rows),'max_abs',max(r['max_abs'] for r in rows))


if __name__=='__main__':main()
