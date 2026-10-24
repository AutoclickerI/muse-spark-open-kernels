"""Check the admitted LOP3 cell's binding API and force its Hopper launch glue."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from check_moe_nt1_fixed import Portable,load


def main():
    root=Path(__file__).resolve().parents[2]
    cell_path='problems/muse_decode_layer/dev/nt1_branches/nt1_lop3_cell.py'
    c=load(cell_path,'handoff_cell');m=load('prof/lab/moe_nt1_fixed.py','handoff_oracle')
    assert c.NT1_SOURCES['lop3']==m.NT1_SOURCES['INC']
    w=m.rg_generate_weights(5353);w['post_norm_w']=w['post_expert_norm']
    state=c.prepare_kind(w,'lop3')
    bank=dict(state['banks']['lop3'],cnt=state['counters']['lop3'],gs13=state['gs13'],gs2=state['gs2'],pnw=state['pnw'],grid=state['sms'])
    bound=c.lop3_bind(bank)
    assert bound['banks']['lop3']['w13p'].data_ptr()==bank['w13p'].data_ptr()
    native=c.nt1_module('lop3');checks=forced=0;worst=0.
    for rows in (1,2,7,8,16,31,32):
        ds=[m.fx_make_fixture(w,rows,9000+rows+7919*j) for j in range(3)]
        wants=[m.nt1_fp32(d,w) for d in ds]
        graph=m.LabGraph(lambda d:c.lop3_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],bound),ds[0])
        for d,want in zip(ds,wants):
            graph.refresh(d);got=graph.replay();torch.cuda.synchronize();e,r=m.nt1_error(got,want)
            assert e<.01 and r<.05;(worst,checks)=(max(worst,e),checks+1)
        old_flag,old_load,old_modules=c.NT1_HOPPER,c.load_inline,c.NT1_MODULES
        portable=Portable(native,'INC',state['sms'])
        def fake_load(**kw):
            assert '-gencode=arch=compute_90a,code=sm_90a' in kw['extra_cuda_cflags']
            return portable
        c.NT1_HOPPER=True;c.load_inline=fake_load;c.NT1_MODULES={}
        try:
            fake=c.lop3_bind(dict(bank,grid=132))
            graph=m.LabGraph(lambda d:c.lop3_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],fake),ds[0])
            for d,want in zip(ds,wants):
                graph.refresh(d);got=graph.replay();torch.cuda.synchronize();e,r=m.nt1_error(got,want)
                assert e<.01 and r<.05;forced+=1
        finally:c.NT1_HOPPER=old_flag;c.load_inline=old_load;c.NT1_MODULES=old_modules
    result=dict(path=cell_path,sha256=hashlib.sha256((root/cell_path).read_bytes()).hexdigest(),cuda_identical_to_h200_INC=True,
        native_graph_checks=checks,forced_hopper_graph_checks=forced,worst_fp32=worst,aliases_existing_bank=True)
    (root/'runs/screen/nt1_handoff_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
