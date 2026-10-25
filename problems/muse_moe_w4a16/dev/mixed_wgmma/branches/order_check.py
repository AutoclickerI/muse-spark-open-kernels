"""Bit-exact heavy intermediates, shared routing, before the paired order lab."""
import json
import runpy
import time
from pathlib import Path
import torch

H=Path(__file__).resolve().parent


def main():
    m=runpy.run_path(str(H/'order_payload.py'),init_globals={'lab_cold_l2':lambda:None});m['op_build']()
    checks=0
    for seed in range(3):
        w=m['dc_weights'](5353+7919*seed,'cuda');bank=m['op_prepare'](w)
        for size in (127,128,255,256,257):
            d=m['dc_fixtures'](w,size)[0];p=m['op_plan'](d,bank,0);p['steps'][0]()
            h13,h2,nv,ids,tw,pos,tasks=p['owners'];counts=h13[4].cpu().tolist();off=h13[5].cpu().tolist()
            rows=[r for e,n in enumerate(counts) if n>16 for r in range(off[e],off[e]+n)]
            rows=torch.tensor(rows,device='cuda',dtype=torch.long)
            for first in (1,0):
                ptr=h13 if first else h2
                if not first:ptr[3].normal_()
                got=[]
                for a,mod in enumerate(m['OP_MODS']):
                    v=list(ptr);bb=bank['bases'][a];key='w13' if first else 'w2';v[:2]=[bb[key],bb[key+'_sf']]
                    mod.br_run([x.data_ptr() for x in v],26 if first else 21,1056 if size<=128 else 2112,size,torch.cuda.current_stream().cuda_stream)
                    got.append(tuple(t.index_select(0,rows).clone() for t in ((h13[10],) if first else (h13[11],h13[12]))))
                torch.cuda.synchronize()
                assert all(torch.equal(a,b) for arm in got[1:] for a,b in zip(got[0],arm)),(seed,size,first)
                checks+=len(got)-1
        print('SEED',seed,'heavy intermediate bit checks',checks,flush=True)
        del bank,w;torch.cuda.empty_cache()
    result=dict(heavy_intermediate_bit_checks=checks,sizes=[127,128,255,256,257],seeds=3)
    (H/'order_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('\n'.join(m['lab'](time.monotonic()+50,{'device':torch.empty(0,device='cuda')})))


if __name__=='__main__':main()
