"""The production N16G4 nf plan against its measured eea9d87b arm."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import torch
H=Path(__file__).resolve().parent

def main():
    spec=importlib.util.spec_from_file_location('nf_light_integration',H/'s12_light.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ns=runpy.run_path(str(H/'light_payload.py'),init_globals={'lab_cold_l2':lambda:None});ns['op_build']()
    # Compile the candidate nf entry point, then prepare identical packed banks.
    mod.nf_module();rows=[]
    for seed in range(3):
        w=ns['dc_weights'](5353+7919*seed,'cuda');bank=ns['op_prepare'](w)
        for size in (128,256):
            ds=ns['dc_fixtures'](w,size);wants=ns['dc_oracles'](ds,w)
            for fi,(data,want) in enumerate(zip(ds,wants)):
                streams=(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event())
                plan=mod.nf_plan(data,bank['bases'][3],bank['native'],streams)
                for f in plan['steps']:f()
                o=plan['out'].clone();r=ns['op_call'](data,bank,3).clone();torch.cuda.synchronize()
                eq=torch.equal(o,r);err=ns['op_error'](o,want)
                assert eq,(seed,size,fi);assert err[0]<.015 and err[1]<.075,err
                rows.append(dict(seed=seed,m=size,fixture=fi,equal=eq,error=err))
        del bank,w;torch.cuda.empty_cache()
    out=dict(patch_sha256=hashlib.sha256((H/'patch_s12_light.py').read_bytes()).hexdigest(),
             candidate_sha256=hashlib.sha256((H/'s12_light.py').read_bytes()).hexdigest(),rows=rows)
    (H/'light_patch_equiv.json').write_text(json.dumps(out,indent=2)+'\n');print('LIGHT_PATCH_EQUIV',len(rows),sum(r['equal'] for r in rows),flush=True)

if __name__=='__main__':main()
