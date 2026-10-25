"""Compare the new mixed control with the literal accepted8483 payload."""
import hashlib
import json
import runpy
from pathlib import Path
import torch

H=Path(__file__).resolve().parent;R=H.parents[4]


def main():
    src=R/'prof/lab/moe_w13_issue_m256.py'
    assert hashlib.sha256(src.read_bytes()).hexdigest()=='cb20858c5de9bfaca16ee289529cc2f46554ed575e32a7ce16dcccd31b1cd179'
    old=runpy.run_path(str(src));old['wi_prepare'].__globals__['WX_DEADLINE']=float('inf')
    new=runpy.run_path(str(R/'prof/lab/moe_m256_mix.py'));w=new['dc_weights'](5353,'cuda');a=old['wi_prepare'](w);b=new['mx_prepare'](w)
    for key in ('w13','w2'):
        assert torch.equal(a[key+'_rec'],b[key]) and torch.equal(a[key+'_rec_sf'],b[key+'_sf'])
    ds=new['dc_fixtures'](w,256,True)+new['dc_fixtures'](w,256)
    for ids in (torch.full_like(ds[0]['topk_ids'],255),ds[0]['topk_ids'].roll(8,0)):
        ds.append(dict(ds[0],topk_ids=ids))
    ga=new['MWGraph'](lambda d:old['wi_call'](d,a,6,2112),ds[0]);gb=new['MWGraph'](lambda d:new['mx_call'](d,b,(0,32),(0,32),0,2112,2),ds[0]);worst=0.
    for d in ds:
        ga.refresh(d);gb.refresh(d);x=ga.replay().clone();y=gb.replay();torch.cuda.synchronize()
        diff=float((x.float()-y.float()).abs().max());worst=max(worst,diff);assert torch.equal(x,y),diff
    result=dict(accepted_payload_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),new_payload_sha256=hashlib.sha256((R/'prof/lab/moe_m256_mix.py').read_bytes()).hexdigest(),checks=len(ds),bit_exact_output=True,identical_packed_weights_and_scales=True,max_abs=worst)
    (H/'control_agreement.json').write_text(json.dumps(result,indent=2)+'\n');print('MIX_CONTROL_PASS',result)


if __name__=='__main__':main()
