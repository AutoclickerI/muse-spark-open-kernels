"""Force the actual packed WGMMA-producer glue in full-layer fresh-input graphs on SM120."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer')]
from local_eval import load_module,Problem,clone,copy_into,capture


def main():
    ref=load_module('reference',ROOT/'problems/muse_decode_layer/reference.py')
    if torch.cuda.get_device_capability()!=(9,0):ref._fa3_scheduler_metadata=lambda *a,**k:None
    candidate=ROOT/'problems/muse_decode_layer/cand_r3_mixed_wgmma.py'
    mod=load_module('forced_mw_submission',candidate)
    weights=ref.generate_weights(ref.WEIGHTS_SEED)
    state=mod.prepare(weights)
    state['mw_bank']=mod.mw_add_planes(mod.mw_prepare(weights))
    problem=Problem(ref,weights)
    spec=json.loads((ROOT/'upstream/api/problems/muse_decode_layer.json').read_text())
    cases=[b for b in spec['benchmarks'] if b['load'] in ('verify 16','verify 32')]
    sms=torch.cuda.get_device_properties(0).multi_processor_count
    variants=((64,16,128,2,2*sms,True,True),(128,32,128,2,2*sms,True,True))
    results=[]
    for choice in variants:
        state['mw_choices']={128:choice,256:choice}
        for case in cases:
            static=clone(problem.make_input(case,91234))
            graph,out=capture(lambda d:mod.kernel_fn(d,state),static)
            for j in range(3):
                fresh=problem.make_input(case,91234+7919*j)
                copy_into(static,fresh);graph.replay();torch.cuda.synchronize()
                ok,msg=problem.check(static,out)
                row=dict(choice=choice,load=case['load'],layer=case['layer'],seed=91234+7919*j,ok=ok,msg=msg)
                results.append(row)
                print(row,flush=True)
                assert ok,row
            del graph,out,static
    report={'sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'cc':torch.cuda.get_device_capability(),
            'checks':results,'count':len(results)}
    (ROOT/'runs/screen/decode_mixed_wgmma_forced.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
