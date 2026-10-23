"""Testable geometry/roofline hypotheses; no cache-counter measurement implied."""
import json,math
from pathlib import Path
M,N,K,SM=16384,2304,8192,132
flops=2*M*N*K
rows=[]
for bm,bn in ((320,128),(128,320)):
    mt,nt=math.ceil(M/bm),math.ceil(N/bn)
    tiles=mt*nt;waves=math.ceil(tiles/SM)
    tail=M*N/(mt*bm*nt*bn);wave=tiles/(waves*SM)
    traffic=2*M*K*nt+2*N*K*mt+2*M*N
    rows.append(dict(bm=bm,bn=bn,tiles=tiles,waves=waves,tail_efficiency=tail,
                     wave_efficiency=wave,combined=tail*wave,
                     naive_tile_bytes=traffic,naive_hbm_us=traffic/4.8e12*1e6))
report=dict(kind='geometry model, ignoring cooperative CTA geometry; DRAM/L2 counters unmeasured',
            m=M,n=N,k=K,flops=flops,compute_floor_us=flops/989e12*1e6,
            compulsory_bytes=2*(M*K+N*K+M*N),output_store_us=2*M*N/4.8e12*1e6,
            qkv_weights_bytes=2*N*K,rows=rows,
            chunks=[dict(m=c,working_set_bytes=2*(N*K+c*K+c*N)) for c in (512,1024,2048,4096,8192)])
root=Path(__file__).resolve().parents[3]
(root/'runs/screen/prefill_gemm/qkv_geometry.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
