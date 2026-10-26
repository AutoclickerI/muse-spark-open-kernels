"""Retain instruction offsets, including the armed NT1 acquire and bulk copies."""
import json,re
from pathlib import Path
H=Path(__file__).resolve().parent
rows=[]
for p in sorted((H/'native').glob('*.sass')):
    if p.stem.startswith('BASE_') or p.stem.startswith('q3_'):continue
    for chunk in p.read_text().split('Function : ')[1:]:
        name=chunk.splitlines()[0]
        ops=[(a,b.strip()) for a,b in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',chunk)]
        waits=[i for i,(a,b) in enumerate(ops) if 'ACQBULK' in b]
        if not waits:continue
        entries=[]
        for k,i in enumerate(waits):
            start=waits[k-1]+1 if k else 0
            loads=[(a,b) for a,b in ops[start:i] if any(q in b for q in ('LDGSTS','UTMALDG','UBLKCP')) or ('LDG.' in b and '.128' in b)]
            entries.append(dict(wait=ops[i][0],preloads=loads[:6],count=len(loads),global_loads=sum('LDG.' in b for _,b in ops[start:i]),mma=sum('HMMA' in b or 'HGMMA' in b for _,b in ops[start:i])))
        rows.append(dict(source=p.stem,kernel=name,acquires=entries))
for src in ('NF_1_o','NF_8_o','NF_1_qkv3','NF_16_o','P10_pool','NF_16_unpooler','NF_32_qkv_gate','NF_b2o_OLAYER_CUDA','NF_OLAYER_CUDA','NF_MW32'):
    rr=[r for r in rows if r['source']==src];assert rr,src
    assert all(r['acquires'][0]['count']>0 for r in rr),src
for src in ('N13_RVC_NINDEX0_CUDA','N13_RVC_NBASE1_CUDA','N13_RVC_NBASE2_CUDA','ALL_RVC_NBASE2_CUDA'):
    rr=[r for r in rows if r['source']==src and len(r['acquires'])==2];assert rr,src
    # The first ACQBULK belongs to the null-ready prepare path. In the armed
    # path, route metadata is acquired and its weight TMA precedes the second.
    assert all(r['acquires'][1]['count']>0 for r in rr),src
rr=[r for r in rows if r['source']=='B1_GLUE_CL_CUDA' and 'ew_boundary_cl' in r['kernel']]
assert len(rr)==14 and all(r['acquires'][0]['count']>=3 for r in rr)
for src in ('H13_P3B_CUDA','H2_P3B_CUDA','ALL_P3B_CUDA','KV_PARTIAL_CUDA'):
    rr=[r for r in rows if r['source']==src];assert rr,src
    assert all(r['acquires'][0]['count']>0 for r in rr),src
rr=[r for r in rows if r['source']=='GATE'];assert len(rr)==1 and rr[0]['acquires'][0]['global_loads']>0 and rr[0]['acquires'][0]['mma']>0
(H/'sass_order.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS',len(rows),'kernel acquire-placement witnesses')
