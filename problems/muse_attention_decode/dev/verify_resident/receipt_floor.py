"""Recover the chosen sliding loads' critical increments and logical KV floors from the landed receipt."""
import base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
p=R/'runs/muse_prefill_layer/20261025T071342Z_check_c6bd22e2084c4809.json'
def strings(x):
    if isinstance(x,str):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from strings(v)
    elif isinstance(x,list):
        for v in x:yield from strings(v)
receipt=json.loads(p.read_text());blob=next(l.split('TL_XZ85 ',1)[1] for s in strings(receipt) for l in s.splitlines() if 'TL_XZ85 ' in l)
data=json.loads(lzma.decompress(base64.b85decode(blob)));rows=[]
for r in data['results']:
    if r['case'] not in ('v16S','v32S'):continue
    bystage={n:[] for n in data['stages']}
    for n in r['nodes']:bystage[data['stages'][n[1]]].append(n)
    end=max(n[5] for name,nodes in bystage.items() if name in ('qkv_gate','qkv3') for n in nodes)
    attn_end=max(n[5] for n in bystage['attn']);kv=r['footprints']['attn']['kv']
    rows.append(dict(case=r['case'],critical_increment_us=round(attn_end-end,6),KV_logical_bytes=kv,KV_floor_us=kv/4.8e6,
                     total_attention_logical_bytes=r['footprints']['attn']['read'],byte_floor_TB_s=4.8,
                     caveat='Logical unique bytes are a compulsory-byte model, not measured DRAM traffic. Same unchanged sliding path in S14; native new batch remains unmeasured.'))
result=dict(job='c6bd22e2084c4809',receipt=str(p.relative_to(R)),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=rows)
(H/'floor.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
