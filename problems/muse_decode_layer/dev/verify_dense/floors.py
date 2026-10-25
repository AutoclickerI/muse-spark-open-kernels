from pathlib import Path
import json,re,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
log=(ROOT/'problems/muse_moe_w4a16/lab/profiles/lab_verify_streaming_synthetic_moe.19f3a13b17b44b6f.log').read_text()
old={(int(m),role):float(us) for m,role,us in re.findall(r'M=(\d+) role=(\w+) control=\S+ old_us=([\d.]+)',log)}
shapes=dict(qkv_gate=(2304,8192),o=(8192,1024),pool_router=(4608,8192),unpooler=(8192,4096));rows=[];lines=[]
lines+=['M>=64 uses the full unpooler. BF16 bank197,132,288B, not138,412,032B (the latter uses the <=8-token unpooler slice). Router uses4096 pool+256 hi+256 lo rows, all FP32 outputs. Bytes = W+X+Y once; floors use3.6TB/s and800TFLOP/s, without launch/glue overhead.','', '| M | Role | W MB | All MB | GFLOP | Byte/math floor us | Floor us | cuBLAS/Lt us | Gap us/layer; x62 us |','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for m in (64,128,256):
 for role,(n,k) in shapes.items():
  wb=2*n*k;ib=2*m*k;ob=(4 if role=='pool_router' else 2)*m*n;flops=2*m*n*k
  bf=(wb+ib+ob)/3.6e6;mf=flops/800e6;floor=max(bf,mf);base=old.get((m,role));gap=None if base is None else base-floor
  row=dict(m=m,role=role,n=n,k=k,weight_bytes=wb,input_bytes=ib,output_bytes=ob,bytes=wb+ib+ob,flops=flops,byte_floor_us=bf,math_floor_us=mf,floor_us=floor,incumbent_us=base,gap_us=gap,gap_62_layers_us=None if gap is None else gap*62)
  rows.append(row);b='unmeasured' if base is None else f'{base:.3f}';g='pending lab' if gap is None else f'{gap:.2f}; {gap*62:.1f}'
  lines.append(f'| {m} | {role} | {wb/1e6:.2f} | {(wb+ib+ob)/1e6:.2f} | {flops/1e9:.3f} | {bf:.2f}/{mf:.2f} | {floor:.2f} | {b} | {g} |')
 forload=[r for r in rows if r['m']==m];total=sum(r['floor_us'] for r in forload);base=None if m==64 else sum(r['incumbent_us'] for r in forload)
 lines.append(f'| {m} | **sum** | 197.13 | {sum(r["bytes"] for r in forload)/1e6:.2f} | {sum(r["flops"] for r in forload)/1e9:.3f} | — | **{total:.2f}** | '+('unmeasured | pending lab |' if base is None else f'**{base:.3f}** | **{base-total:.2f}; {(base-total)*62:.1f}** |'))
s=json.loads((ROOT/'problems/muse_decode_layer/dev/best7/sensitivity.json').read_text())
gain=1.
for m,load in ((128,'verify 16'),(256,'verify 32')):
 rr=[r for r in rows if r['m']==m];T=s['loads'][load]['us'];saving=sum(r['gap_us'] for r in rr)*62;gain*=(T/(T-saving))**(1/6)
lines+=['',f'At the FOCUS best6 denominators, reaching these floors at verify16/32 gives{(gain-1)*100:.2f}% overall. This is optimistic: the floor excludes launch overhead and assumes stage peak rates;1.3x these floors gives '+f'{(math.prod((s["loads"][load]["us"]/(s["loads"][load]["us"]-(sum(r["incumbent_us"] for r in rows if r["m"]==m)-1.3*sum(r["floor_us"] for r in rows if r["m"]==m))*62))**(1/6) for m,load in ((128,"verify 16"),(256,"verify 32")))-1)*100:.2f}%.' ,'M64 has no incumbent measurement in this receipt; the batched lab supplies it. No number is interpolated.']
(HERE/'floor_table.md').write_text('\n'.join(lines)+'\n');(HERE/'floors.json').write_text(json.dumps(dict(rows=rows,overall_floor_ceiling_pct=(gain-1)*100),indent=2)+'\n');print('\n'.join(lines))
