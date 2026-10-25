"""Exercise the actual frozen decode-cell API with fresh captured routes."""
import hashlib
import json
import runpy
from pathlib import Path
import torch

H=Path(__file__).resolve().parent;R=H.parents[4]


def main():
    helpers=runpy.run_path(str(R/'prof/lab/moe_m256_mix.py'));cell=runpy.run_path(str(H/'unmeasured_cell.py'))
    w=helpers['dc_weights'](29101,'cuda');bank=cell['m256_mix_prepare'](w)
    ds=helpers['dc_fixtures'](w,256);base=ds[0]
    for n in (0,8,9,16,17,32,33,64,65,119,256):
        ids=torch.full_like(base['topk_ids'],255);ids[:n,0]=0;ds.append(dict(base,hidden_states=torch.randn_like(base['hidden_states']),topk_ids=ids))
    wants=helpers['dc_oracles'](ds,w)
    graph=helpers['MWGraph'](lambda d:cell['m256_mix_experts'](d['hidden_states'],d['topk_weights'],d['topk_ids'],bank),ds[0]);rows=[]
    for d,want in zip(ds,wants):
        graph.refresh(d);out=graph.replay();e,r=helpers['mx_error'](out,want);assert e<=.015 and r<=.075,(e,r);rows.append((e,r))
    shifted=dict(ds[0],topk_ids=ds[0]['topk_ids']+64)
    out=cell['m256_mix_experts'](shifted['hidden_states'],shifted['topk_weights'],shifted['topk_ids'],bank,local_start=64)
    e,r=helpers['mx_error'](out,wants[0]);assert e<=.015 and r<=.075;rows.append((e,r))
    result=dict(sha256=hashlib.sha256((H/'unmeasured_cell.py').read_bytes()).hexdigest(),checks=len(rows),max_L2=max(e for e,r in rows),max_row=max(r for e,r in rows),captured_route_changes=True,local_start64=True)
    (H/'cell_agreement.json').write_text(json.dumps(result,indent=2)+'\n');print('MIX_CELL_PASS',result)


if __name__=='__main__':main()
