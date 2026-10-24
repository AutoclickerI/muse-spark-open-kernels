"""Rebuild the receipt-derived stage lever table (no judge access)."""
from pathlib import Path
import sys, json, re
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from sim import bound, problems, muse

SOURCES=[('V9','cand_r3_v9.0e94268cf5654012'),('cost','cand_r3_v9_cost_v9.906ca5580aef4f86')]

def stages(report):
    result=dict(P=0.,A=0.,M=0.,G=0.)
    partial=bool(re.search(r'^\.\.\. \d+ more$',report,re.M))
    for line in report.splitlines():
        match=re.match(r'^(.*?)\s+([\d.]+) µs\s+[\d.]+%\s+\d+\s*$',line)
        if not match:continue
        name,us=match.group(1),float(match.group(2))
        if 'moe_fused' in name:key='M'
        elif any(s in name for s in ('_gemv_kernel','nvjet_','cublasLt::splitKreduce')):key='P'
        elif any(s in name for s in ('attn_main','attn_combine','flash::','_qk_prep_kernel','_norm_gate_kernel')):key='A'
        elif any(s in name for s in ('_boundary_kernel','_route_norm_kernel','vectorized_elementwise')):key='G'
        else:raise ValueError(name)
        result[key]+=us
    total=float(re.search(r'^total\s+([\d.]+) µs',report,re.M)[1])
    result['unreported']=max(0.,total-sum(result.values())) if partial else 0.
    return result,partial

def floors(case,tf):
    b=dict(case);b.setdefault('q_len',1);t=b['batch']*b['q_len']
    touched,routes=bound.inferred_routing('muse_decode_layer',b,tf)
    # Ideal fused IO: do not charge intermediate spills or launches as compulsory.
    p=sum((bound.projection_work(dict(m=t,layer=n),activation_factor=0) for n in bound.PROJ),bound.Work())
    context=b['ctx']*(.925 if b.get('spread')=='agentic' else 1.)
    a=bound.attention_work(b['batch'],b['q_len'],context,b['layer'],external_io=False)
    m=bound.nvfp4_expert_work(touched,routes,t,external_io=False)
    return {k:bound.floor(w,bound.SPEC_BW,replays=0).adjusted_us for k,w in [('P',p),('A',a),('M',m)]}

def main():
    rows=[]
    lines=['## H200 stage levers: V9 and cost receipts','',
        'All values are microseconds. Each P/A/M cell is **profile time / modeled floor**; '
        'glue is boundary + routing + remaining elementwise kernels. V9 = receipt '
        '`0e94268cf5654012`; cost = `906ca5580aef4f86`. Source profiles are linked below. '
        'These are separate router seeds: compare scores to each run’s own baseline, '
        'rather than treating differences in absolute MoE time as kernel gains.','',
        'Floors use one read of stored weights/cache, 4.8 TB/s HBM, 989 TF/s dense BF16, '
        '1979 TF/s FP8 QK and the audited NVFP4 issue floor from `sim/bound.py`. '
        'Routing counts are inferred expected counts from the official layer floor, '
        'not exact counts recovered from these seeds. Full agentic attention uses its '
        'guaranteed 92.5% context minimum. No launch/PDL wait is charged to the floor. '
        'Per-stage floors cannot be summed into a whole-layer bound when resources overlap.','',
        '| Load / context | Run | Whole layer | Projections / floor | Attention / floor | MoE / floor | Glue |','|---|---|---:|---:|---:|---:|---:|']
    for tag,stem in SOURCES:
        data=json.loads((ROOT/'problems/muse_decode_layer/profiles'/f'{stem}.json').read_text())
        for profile,bench,tf in zip(data['profiles'],data['benchmarks'],problems.task_floors('muse_decode_layer')):
            c=profile['case'];s,partial=stages(profile['report']);f=floors(c,tf)
            rows.append(dict(receipt=stem,case=c,profile=s,floors=f,partial=partial,layer_us=bench['median_ns']/1000))
    for index in range(12):
        for row in (rows[index],rows[12+index]):
            c=row['case'];s=row['profile'];f=row['floors'];partial=row['partial']
            cells=[f'{">=" if partial and k in ("P","A") else ""}{s[k]:.1f} / {f[k]:.2f}' for k in ('P','A','M')]
            glue=f'{s["G"]:.1f}'+(f' + {s["unreported"]:.1f} unreported' if partial else '')
            lines.append(f'| {c["load"]} {c["layer"]} | {"V9" if row in rows[:12] else "cost"} | {row["layer_us"]:.1f} | '+ ' | '.join(cells)+f' | {glue} |')
    lines += ['',
        'P includes cuBLAS split-K reductions; A includes QK preparation, cache writes, FA3 scheduling '
        'and norm/gate kernels. Profile rows include PDL wait/overlap, so their sum can exceed '
        'the whole-layer time. The two cost verify-1 profiles truncate three small rows: their '
        'P/A entries are lower bounds and the unreported residual is shown explicitly, rather '
        'than assigned to a stage without evidence. Rounding of report rows is about 0.1 us.','',
        'Largest levers: verify-32 MoE is 443–477 us; full attention (including preparation) is 389–390 us. '
        'At decode-32, full attention is 226–229 us. Small-load projection time remains '
        'roughly 45–66 us against a 29–41 us floor, while boundary/routing glue costs '
        'about 12–27 us. These justify packing mixed-input/reuse MoE, attention split-K '
        'and a few projection recipes into the same timed portfolio.','',
        '[V9 profile](../profiles/cand_r3_v9.0e94268cf5654012.json) · '
        '[cost profile](../profiles/cand_r3_v9_cost_v9.906ca5580aef4f86.json) · '
        '[rebuild script](omni/lever_table.py)','']
    doc=ROOT/'problems/muse_decode_layer/dev/NEXT_CHECKS.md'
    text=doc.read_text();marker='## H200 stage levers: V9 and cost receipts'
    if marker in text:text=text[:text.index(marker)].rstrip()+'\n\n'
    doc.write_text(text+'\n'.join(lines))
    (Path(__file__).parent/'lever_table.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
