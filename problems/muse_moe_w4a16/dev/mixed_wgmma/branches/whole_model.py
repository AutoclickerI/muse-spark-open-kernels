"""Whole-MoE model from 3e8996f8 phase times. Bounds, not H200 predictions.

Logical repeated bytes can hit L2: no DRAM-counter claims. The high-MLP stream
rate belongs to each phase's address mix. Scaling it to multicast is an
optimistic bound, not a measured multicast throughput.
"""
import json
from pathlib import Path

H=Path(__file__).resolve().parent
DATA={128:dict(base=196.44,heavy=[(56.43,50.62,38.13,29.18,81.79,129.50,170.74),(37.11,32.90,25.06,17.36,40.89,64.75,85.37)],light=[(74.85,69.89,64.96,61.39,229.64,271.06,12.02),(44.52,40.73,35.99,33.87,114.82,135.53,12.00)]),
      256:dict(base=270.64,heavy=[(109.47,94.43,70.17,50.32,130.55,282.07,402.74),(60.66,54.62,39.53,26.75,65.27,141.03,201.37)],light=[(78.42,73.70,68.96,64.40,241.70,286.26,11.79),(46.22,42.50,37.80,35.36,120.85,143.13,11.89)])}

def main():
    out={}
    for m,d in DATA.items():
        heavy=d['heavy'];light=d['light'];rows=heavy+light
        unique=sum(p[4] for p in rows)/4.58
        # Preserve the measured non-load/non-decode residual. Optimistically
        # scale staging with logical bytes and dequant with weight multiplicity.
        def once(cluster):
            return sum(p[2]*(p[4]+p[6]/cluster)/(p[5]+p[6])+(p[1]-p[2])+(p[0]-p[1])*p[4]/p[5] for p in heavy)
        h=sum(p[0] for p in heavy)
        # New shared scheduler: light <=16 reads once, heavy N32 unchanged.
        # Give shared work both the full-rate byte stream and measured math/
        # staging residue. Hide 50% of residue (explicit design assumption).
        stream=sum((p[5]+p[6])/((p[5]+p[6])/p[3]) for p in heavy)
        stream+=sum((p[4]+p[6])/((p[5]+p[6])/p[3]) for p in light)
        residual=sum(p[0]-p[3] for p in rows)
        overhead=15. # measured route + F4 finish, rounded
        # Streams are a resource sum, never max(heavy,light). The range makes
        # overlap assumptions visible; 0.5 is the planning point, not a receipt.
        q=[stream+hide*residual+overhead for hide in (0.,.5,1.)]
        out[m]=dict(unique_MB=sum(p[4] for p in rows),unique_stream_us=unique,target_us=1.3*unique,
                    current_us=d['base'],heavy_observed_us=h,
                    decode_once_no_multicast_us=d['base']-h+once(1),
                    decode_once_multicast4_optimistic_us=d['base']-h+once(4),
                    shared_queue_stream_service_us=stream,shared_queue_residual_us=residual,
                    shared_queue_all_half_none_residual_us=q)
    (H/'whole_model.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
