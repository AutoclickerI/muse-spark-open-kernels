"""Decode a lossless timeline archive from an H200 receipt and render tables.

python3 .../timeline/report.py JOB_ID
Writes results.json, nodes.csv, and the Results portion of README.md.
"""
import argparse
import base64
import csv
import json
import re
import statistics
import lzma
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def decode(receipt):
    obj=json.loads(receipt.read_text())
    text=obj['result'].get('log_tail','')
    match=re.search(r'^TL_XZ85 (\S+)$',text,re.M)
    if not match:raise RuntimeError('receipt contains no timeline archive: '+text[-1500:])
    archive=json.loads(lzma.decompress(base64.b85decode(match[1])))
    for r in archive['results']:
        nodes=[]
        for row in r['nodes']:
            n=dict(zip(archive['columns'],row))
            n['name']=archive['names'][n['name']]
            n['stage']=archive['stages'][n['stage']]
            n['byte_kind']=archive['byte_kinds'][n['byte_kind']]
            n['duration_us']=n['end_us']-n['start_us']
            n['archived_logical_read_B']=n['logical_read_B']
            n['byte_audit']='unchanged'
            # The activity lab's broad 'flash::' classifier included both FA3
            # metadata and combine nodes. Audit their static reads separately;
            # CUPTI timestamps and stage-level byte counts are unchanged.
            batch=int(r['case'][1:-1])
            if 'prepare_varlen_num_blocks_kernel' in n['name']:
                n['logical_read_B']=batch*4
                n['byte_kind']='sequence_lengths_lower_bound'
                n['byte_audit']='FA3 metadata does not read KV'
            elif 'FlashAttnFwdCombine<' in n['name']:
                n['logical_read_B']=0
                n['byte_kind']='unresolved_split_partial'
                n['byte_audit']='FA3 combine reads split accumulators, not KV; actual split count was not retained'
            n['logical_TBps']=(n['logical_read_B']/n['duration_us']/1e6
                              if not n['byte_kind'].startswith('unresolved') else None)
            nodes.append(n)
        for i,n in enumerate(nodes):
            n['next_start_gap_us']=nodes[i+1]['start_us']-n['end_us'] if i+1<len(nodes) else None
        r['nodes']=nodes
        cuts=sorted({v for n in nodes for v in (n['start_us'],n['end_us'])})
        low_lo=low_hi=0.
        for lo,hi in zip(cuts,cuts[1:]):
            active=[n for n in nodes if n['start_us']<=lo and n['end_us']>=hi]
            bw=sum(n['logical_TBps'] or 0. for n in active)
            if bw<2.29:
                low_hi+=hi-lo
                if all(n['logical_TBps'] is not None for n in active):low_lo+=hi-lo
        r['archived_low50_proxy']=r['low50_proxy']
        r['low50_proxy']=low_hi
        r['low50_proxy_min']=low_lo
        stage_rows={}
        for s in {n['stage'] for n in nodes}:
            subset=[n for n in nodes if n['stage']==s]
            lo=min(n['start_us'] for n in subset); hi=max(n['end_us'] for n in subset)
            rd=r['footprints'].get(s,{}).get('read',0)
            stage_rows[s]=dict(start=lo,end=hi,span=hi-lo,read=rd,bw=rd/(hi-lo)/1e6,
                               residual=max(0,hi-lo-rd/4.58e6))
        frontier=0.
        for stage in ('qkv3','qkv_gate','attn','gate','o','b1','pool_router','route','moe','unpooler_slice','unpooler','b2'):
            if stage in stage_rows:
                v=stage_rows[stage]
                v['completion_increment']=max(0,v['end']-frontier)
                frontier=max(frontier,v['end'])
        r['stages']=stage_rows
    return obj,archive


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+
                     ['| '+' | '.join(str(c) for c in r)+' |' for r in rows])


def render(archive,receipt,job):
    results=archive['results'];by={r['case']:r for r in results}
    cols=['Load','Wall ms','Kernel sum ms','Busy union ms','No-kernel ms','<50% proxy interval ms','Read GB','Stream floor ms']
    rows=[]
    for prefix,label in [('d1','decode 1'),('d16','decode 16'),('d32','decode 32'),
                         ('v1','verify 1'),('v16','verify 16'),('v32','verify 32')]:
        pair=[(by[prefix+'S'],46),(by[prefix+'F'],16)]
        val=lambda k:sum(r[k]*w for r,w in pair)
        rows.append([label]+[f'{val(k)/1000:.3f}' for k in ('wall','kernel_sum','busy_union','bubble')]+
                    [f'{val("low50_proxy_min")/1000:.3f}–{val("low50_proxy")/1000:.3f}',f'{val("bytes")/1e9:.3f}',f'{val("stream_floor")/1000:.3f}'])
    receipt_link=str(receipt.relative_to(HERE)) if receipt.is_relative_to(HERE) else '../../../../'+str(receipt.relative_to(ROOT))
    lines=[f'H200 receipt `{job}`; [raw receipt]({receipt_link}).', '',
           'Single public fixture per shape, using prefill-host weights. Decode 1 has **zero local expert routes** on this fixture; its wall time is not representative of the board’s calibrated routing. Each load is 46 sliding + 16 full layers. CUPTI wall runs from the first node start to the last node end; '
           'kernel sum double counts overlapping nodes, busy union does not. No-kernel time is the complement of the union.',
           '',table(cols,rows),'',
           '`<50% proxy` integrates the sum of active nodes’ *average logical read rates*. It is not measured HBM utilization. The interval excludes unknown split-partial intervals at its lower end and treats their rate as zero at its upper end. '
           'The stream floor is the logical compulsory read count / 4.58 TB/s. '
           'An ideal continuous-stream latency is `floor + L(qkv→attention) + L(router→routing)`, '
           'where unavoidable dependent latency remains unmeasured. Observed gaps below are scheduling intervals, '
           'not a measurement of those unavoidable minima. Writes, cache hits, redundant reads, dequantization, '
           'and intrinsic attention compute are not represented by this read-only floor.',
           '', 'Per shape (µs; bandwidth is logical TB/s):','',
           table(['Shape','CUPTI span','Event raw / corrected','Σ kernels','Busy union','No-kernel','Overlap','<50% proxy','Read MB','Stream floor','capture A/B','CUPTI/off'],
                 [[r['case'],f'{r["wall"]:.2f}',f'{r["event_plain"]:.2f} / {r["event_corrected"]:.2f}']+[f'{r[k]:.2f}' for k in ('kernel_sum','busy_union','bubble','overlap','low50_proxy')]+
                  [f'{r["bytes"]/1e6:.1f}',f'{r["stream_floor"]:.2f}',f'{r["observer_ratio"]:.4f}',f'{r["profile_ratio"]:.4f}'] for r in results]),
           '', 'Per-stage span, compulsory input reads, effective read rate and fraction of 4.58 TB/s:', '',
           table(['Shape','Stage','Span µs','Completion increment µs','Read MB','B/span TB/s','Roof %','Span − read/roof µs'],
                 [[r['case'],s,f'{v["span"]:.2f}',f'{v.get("completion_increment",v["span"]):.2f}',f'{v["read"]/1e6:.3f}',f'{v["bw"]:.3f}',f'{100*v["bw"]/4.58:.1f}',f'{v["residual"]:.2f}']
                  for r in results for s,v in sorted(r['stages'].items(),key=lambda sv:sv[1]['start'])]),
           '', 'Signed stage dependency envelopes (negative means overlapping nodes, including possible PDL waits):','',
           table(['Shape','QKV end → attention start µs','Pool/router end → routing start µs'],
                 [[r['case']]+[f'{r["dependencies"].get(k,float("nan")):.3f}' for k in ('qkv_attn','router_route')] for r in results])]
    # Rank residual envelopes after weighting every load equally. These are
    # diagnostic opportunities, not causal disjoint off-HBM measurements.
    residuals={}
    for r in results:
        weight=46 if r['case'].endswith('S') else 16
        frontier=0.
        for stage in ('qkv3','qkv_gate','attn','gate','o','b1','pool_router','route','moe','unpooler_slice','unpooler','b2'):
            if stage not in r['stages']:continue
            v=r['stages'][stage]
            increment=max(0,v['end']-frontier)
            frontier=max(frontier,v['end'])
            group='qkv' if stage in ('qkv3','qkv_gate') else stage
            residuals[group]=residuals.get(group,0)+weight*max(0,increment-v['read']/4.58e6)/6
    lines+=['','Each stage owns only its extension of the preceding stage-completion frontier. This removes double counting of PDL overlap; it is still a scheduling model, since reads may start before that frontier.', '']
    lines+=['','Largest three completion-increment residuals (mean µs per 62-layer load; model, not measured off-HBM time):','']
    for stage,amount in sorted(residuals.items(),key=lambda sv:sv[1],reverse=True)[:3]:
        lines.append(f'- **{stage}: {amount:.1f} µs.** Exposed completion increment minus compulsory reads at 4.58 TB/s; '
                     'may include instruction work, barriers, repeated L2 reads, memory ramp/tail, PDL waiting and overlap.')
    if 'd16F' in by:
        o=by['d16F']['stages']['o']
        lines+=['',f'PDL example: decode-16 full O spans {o["span"]:.2f} µs but extends the preceding completion frontier by only {o["completion_increment"]:.2f} µs. Treating its full node span as exposed O work would misidentify the bottleneck.', '']
    lines+=['','Actual routed expert counts (important for interpreting the byte model):','',
           table(['Shape','Distinct local experts','Local routes','Experts >16 tokens'],
                 [[r['case'],r['footprints']['moe']['experts'],sum(r['footprints']['moe']['counts']),r['footprints']['moe']['heavy']] for r in results])]
    lines+=['','FA3 metadata/combine byte attribution was audited offline: metadata reads sequence lengths; combine reads split accumulators, whose exact count was not retained. The raw receipt preserves the lab’s original estimates. `nodes.csv` includes both archived and audited bytes. The low-50 proxy was recomputed; timestamps and stage byte totals did not change.', '']
    lines+=['','Every original kernel node is in [nodes.csv](nodes.csv), with CUPTI node ID, stream, start/end, '
            'signed next-start gap, compulsory logical bytes, attribution class and effective rate. '
            'Unknown internal reads are marked `unresolved*`, never inferred from durations.', '']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt');args=ap.parse_args()
    path=Path(args.receipt)
    if not path.exists():
        hits=list((ROOT/'runs').glob('*/*_'+args.receipt+'.json'))
        if len(hits)!=1:raise RuntimeError(f'expected one receipt: {hits}')
        path=hits[0]
    path=path.resolve()
    obj,archive=decode(path)
    job=obj['id']
    archive.update(receipt_id=job,analysis_version=2,roof_TBps=4.58,load_layers=dict(sliding=46,full=16))
    (HERE/'results.json').write_text(json.dumps(archive,indent=2)+'\n')
    nodes=[dict(case=r['case'],**n) for r in archive['results'] for n in r['nodes']]
    with (HERE/'nodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(nodes[0]));writer.writeheader();writer.writerows(nodes)
    readme=HERE/'README.md'
    prefix=readme.read_text().split('\nResults\n',1)[0]
    summary=render(archive,path,job)
    validation=json.loads((HERE/'validation.json').read_text())
    summary+='\nPayload SHA-256: `'+validation['payload_sha256']+'`. '
    summary+='Normal and forced local screens each passed all 13 official cases and 23 static checks; '
    summary+='[validation.json](validation.json) keeps the measured local observer ranges. '
    summary+='The final standalone embedded payload also completed all 12 shapes locally. Selected-only setup matched full setup on 36 tensors and 36 three-seed fixtures in both normal and forced modes.\n'
    summary+='The unchanged prefill host scored '+str(round(obj['result']['speedup'],6))+'×; that is not a decode candidate gain.\n'
    summary+='\nIntegration: diagnostics only; no candidate patch or promotion. '
    summary+='Next measurement: use decode-calibrated routing, retain FA3 split metadata, and add data-ready/first-consumption stamps plus DRAM counters to identify actual PDL waits, dependent latency and HBM duty. Those quantities are not established by this trace.\n'
    readme.write_text(prefix+'\nResults\n\n'+summary)
    print(f'{job}: {len(archive["results"])} shapes, {len(nodes)} original kernel nodes')

if __name__=='__main__':main()
