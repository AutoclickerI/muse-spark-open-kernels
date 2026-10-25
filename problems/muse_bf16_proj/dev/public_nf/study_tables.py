"""Reproduce factual tables from public result data and our receipt. No external code imports.
Usage: python study_tables.py /path/to/NathenFernandes_muse-bf16-proj/repo
"""
import csv,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
src=Path(sys.argv[1])
ours=H.parents[3]/'runs/muse_bf16_proj/20261025T033426Z_check_be2631409fd64cde.json'
# Paths: dev/public_nf -> dev -> muse_bf16_proj -> problems -> repo
ours=H.parents[3]/'runs/muse_bf16_proj/20261025T033426Z_check_be2631409fd64cde.json'
rs=json.loads(ours.read_text())['result']['records']
us={(r['case']['m'],r['case']['layer']):r['baseline_ns']/r['speedup']/1000 for r in rs if r['event']=='benchmark'}
# Shorthand facts transcribed from final dispatch; WR/WK/KU/S, MT/RT/ST are defined in README.
rows={
('qkv_gate',1):('Q1 D9','1/4/8/4; 1/1/2','R+P+D9',1.5),
('qkv_gate',8):('Q8 D5','1/4/8/4; 1/1/2','R+P+D5',1.5),
('qkv_gate',16):('Q16 push','1/4/8/4; 2/1/1','R+P+D5+cap6',1.5),
('qkv_gate',32):('Q32 push','1/4/8/4; 4/1/1','R+P+D5+cap5',1.5),
('qkv_gate',128):('Qbig SS','64,36,128,3,8,539,5','T+P; raw BF16',2),
('qkv_gate',256):('LT','layout0; [66,124,1,0,0,0,35,0,2]','L; raw BF16',2),
('o',1):('smallm2','4/8/2/1; 1/1/2; opt7','R+WK; 1024 threads',1.5),
('o',8):('v18s','1/8/2/1; 1/1/2; opt6','R+WK; 256 threads',1.5),
('o',16):('m16trace early','2/4/4/1; 2/1/1; XM7','R+WK; early load',1.5),
('o',32):('ol direct','BN32 BK128 ST8 PIPE1 PACK1 HINT1 TW1','T; direct store; raw BF16',2),
('o',128):('ol projection','BN128 BK128 ST4 PIPE3 PACK1 HINT1 TW1','T; producer WG; raw BF16',2),
('o',256):('rl early','BN256 BK64 ST5 PIPE7 PACK1 HINT1 TW1','T; producer WG; raw BF16',2),
('pooler',1):('v18s p32','2/16/8/1; 1/1/1; opt6','R+WK; 1024 threads',1.5),
('pooler',8):('smallm2','2/16/8/1; 1/1/1; opt6','R+WK; 1024 threads',1.5),
('pooler',16):('decodediet cpssm','1/8/16/1; 2/1/2; CP2','R+D9; async weight ring',1.5),
('pooler',32):('bundle2 m32x','1/8/16/1; 4/1/1; XM7 CP1','R+D9; async weight ring; cap2',1.5),
('pooler',128):('LT','layout2; [66,15,1,0,0,4,35,0,2]','L; raw BF16',2),
('pooler',256):('LT','layout3; [66,18,1,0,0,0,35,0,2]','L; raw BF16',2),
('unpooler',1):('smtrace1','1/8/2/4; 1/1/2','R+P; prefix 1024',1.5),
('unpooler',8):('smtrace8','1/16/2/2; 1/1/2','R+P; prefix 1024',1.5),
('unpooler',16):('decodediet cpssm','2/4/16/1; 2/1/2; CP2','R+D9; async weight ring',1.5),
('unpooler',32):('decodediet m32w','BN32 R64 WK4 PF2 Q2 CAP1; split1','W; RS-WGMMA + TMA',1.5),
('unpooler',128):('LT','layout2; [66,18,1,0,0,3,35,0,5]','L; raw BF16',2),
('unpooler',256):('LT','layout3; [66,20,1,0,0,4,35,0,2]','L; raw BF16',2),
('router',1):('rr scalar','M1 NT512 ST2','F; FP32 FMA',4),
('router',8):('routertail local4','MT1 S16 WK8 KU1','R+P; BF16 hi+lo',4),
('router',16):('routertail local4','MT2 S16 WK8 KU1','R+P; BF16 hi+lo',4),
('router',32):('rr SS','BN32 S16 ST4 SW1 RED2 PIPE1','T+P; BF16 hi+lo',4),
('router',128):('routerwide128','BN64 S8 ST8 SW1 RED2 PIPE1; NR64 TM64','T+P; BF16 hi+lo',4),
('router',256):('routerwide256','BN64 S4 ST8 SW0 RED2 PIPE3 SWZ1 HINT0; NR64 TM64','T+P; BF16 hi+lo',4),
}
shape={'qkv_gate':(2304,8192),'o':(8192,1024),'pooler':(4096,8192),'unpooler':(8192,4096),'router':(256,8192)}
public={(int(r['m']),r['layer']):r for r in csv.DictReader((src/'results/v18_shapes.csv').open())}
out=[]
for layer in shape:
 for m in (1,8,16,32,128,256):
  kernel,cfg,mechanism,bpw=rows[layer,m];n,k=shape[layer]
  if layer=='unpooler' and m<=8:n=1024
  wbytes=int(bpw*n*k);bytes_io=2*m*k+(4 if layer=='router' else 2)*m*n
  nf=float(public[m,layer]['solution_us']);our=us[m,layer]
  out.append(dict(layer=layer,m=m,kernel=kernel,config=cfg,mechanism=mechanism,weight_bytes=wbytes,logical_bytes=wbytes+bytes_io,nf_us=nf,our_proj_us=our,nf_over_ours=nf/our,effective_TBs=(wbytes+bytes_io)/nf/1e6))
(H/'shape_study.json').write_text(json.dumps(out,indent=2)+'\n')
with (H/'shape_study.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=out[0].keys());writer.writeheader();writer.writerows(out)
md=['| Projection | t | NF selected kernel/config | NF µs | Our proj µs | Logical MB | Effective TB/s | Mechanism |','|---|---:|---|---:|---:|---:|---:|---|']
for r in out:
 md.append(f"| {r['layer']} | {r['m']} | {r['kernel']}: `{r['config']}` | {r['nf_us']:.2f} | {r['our_proj_us']:.2f} | {r['logical_bytes']/1e6:.2f} | {r['effective_TBs']:.2f} | {r['mechanism']} |")
(H/'shape_table.md').write_text('\n'.join(md)+'\n')
print('\n'.join(md))
