"""Freeze each future N/T point with its exact shape-dispatched S15 patch."""
import importlib.util
import json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
spec=importlib.util.spec_from_file_location('p7_parent_handoff',P/'handoff.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)

def main():
    x=json.loads((H/'h200_result.json').read_text());base=(parent.R/parent.TARGET).read_text()
    runtime=(H/'runtime_generated.py.inc').read_text();out=H/'out/point_handoffs';out.mkdir(parents=True,exist_ok=True)
    points=[]
    for p in x['nominations']:
        points.append(dict(p,**parent.write_piece(base,{p['shape']:p['arm']},p['shape']+'_'+p['arm'],out,runtime if p['arm']=='T' else None)))
    (H/'point_nominations.json').write_text(json.dumps(dict(receipt=x['receipt'],nominations=points),indent=2)+'\n')
    print('P4 exact N/T shape pieces',len(points))
if __name__=='__main__':main()
