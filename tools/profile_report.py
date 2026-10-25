#!/usr/bin/env python3
"""Audit H200 receipts and their *truncated* GPU profiles; host Python, stdlib only.

python3 tools/profile_report.py [all|muse_moe_w4a16] [--serving] [--json]

Paired graph medians and profiler kernel sums are deliberately separate. Bounds are
models, never counters. Names omitted by the judge cannot be reconstructed. A kernel
gets an individual bound only when its work is identifiable; component bounds handle
many-to-one names without allocating bytes in proportion to measured time.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim import hw, muse  # noqa: E402; these modules use only the stdlib
from sim.__main__ import kernels as design_kernels  # noqa: E402
from sim.problems import attention_work, moe_work, proj_work, shape_bounds, spec, task_floors  # noqa: E402

PROBLEMS = ('muse_bf16_proj', 'muse_attention_decode', 'muse_moe_w4a16',
            'muse_decode_layer', 'muse_prefill_layer')
ROW = re.compile(r'^(.*?)\s+([\d,.]+)\s*(?:µs|us)\s+([\d.]+)%\s+(\d+)\s*$')
TOTAL = re.compile(r'^total\s+([\d,.]+)\s*(?:µs|us)', re.M)
BUSY = re.compile(r'^GPU busy.*?([\d,.]+)\s*(?:µs|us)', re.M)


def case_key(c):
    return tuple(sorted((k, v) for k, v in c.items() if k not in ('seed', 'weight', 'repeats')))


def parse_profile(report):
    """Rows in the saved report are rounded to 0.1 us; preserve the omitted mass."""
    rows = []
    for line in report.splitlines():
        m = ROW.match(line)
        if m:
            rows.append(dict(name=m[1].strip(), us=float(m[2].replace(',', '')),
                             reported_share=float(m[3]), calls=int(m[4])))
    total = TOTAL.search(report)
    if not total:
        raise ValueError('profile has no total')
    total = float(total[1].replace(',', ''))
    missing = re.search(r'^\.\.\. (\d+) more', report, re.M)
    residual = total - sum(r['us'] for r in rows)
    if missing:
        rows.append(dict(name=f"[omitted: {missing[1]} kernel names]", us=max(0.0, residual),
                         reported_share=None, calls=None))
    elif abs(residual) > max(0.15, len(rows) * 0.051):
        raise ValueError(f'profile row sum differs by {residual:g} us')
    busy = BUSY.search(report)
    return dict(rows=rows, total_us=total, busy_us=float(busy[1].replace(',', '')) if busy else total,
                truncated=bool(missing))


def group(name):
    n = name.lower()
    if n.startswith('[omitted:'):
        return 'unattributed'
    if 'moe_fused' in n:
        return 'moe'  # includes route/finalize glue: not additive to a separate glue model
    if 'marlin<' in n or 'moe_gemm_kernel' in n or 'indexed_gemm' in n:
        return 'moe'
    if 'attn_main' in n:
        return 'attention'  # includes QK prep and final norm
    if 'flash::' in n or '_attn_' in n or '_qk_prep' in n or '_norm_gate' in n:
        return 'attention'
    if any(s in n for s in ('nvjet_', 'cublas', '_gemv_kernel', 'skinny_kernel', 'gemm', 'dot_kernel')):
        return 'dense'
    return 'glue'


def practical_work(nbytes, flops):
    mem = hw.stream_us(nbytes)
    comp = flops / (hw.BF16_FLOPS * 0.75) * 1e6
    return dict(bytes=nbytes, flops=flops, memory_us=mem, compute_us=comp,
                work_us=max(mem, comp), limiter='compute' if comp > mem else 'memory')


def component_work(problem, b, bounds):
    tokens = b.get('bs') or b.get('m') or b.get('batch', 1) * b.get('q_len', 1)
    if problem == 'muse_bf16_proj':
        return {'dense': practical_work(*proj_work(b))}
    if problem == 'muse_attention_decode':
        return {'attention': practical_work(*attention_work(b, 1.0))}
    if problem == 'muse_moe_w4a16':
        return {'moe': practical_work(*moe_work(tokens, bounds['touched'], bounds['routes']))}
    dense = [proj_work(dict(m=tokens, layer=layer)) for layer in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')]
    # sim.layer_work counts only old context. For prefill include new causal keys;
    # otherwise the first full chunk has zero attention work despite doing attention.
    attn_b = dict(b)
    q = b.get('q_len', 1)
    if q > 8:
        if b['layer'] == 'full':
            attn_b['ctx'] = b['ctx'] + (q + 1) / 2
        else:
            attn_b['ctx'] = sum(min(muse.WINDOW, b['ctx'] + j + 1) for j in range(q)) / q
    return {'dense': practical_work(sum(v[0] for v in dense), sum(v[1] for v in dense)),
            'attention': practical_work(*attention_work(attn_b, 1.0)),
            'moe': practical_work(*moe_work(tokens, bounds['touched'], bounds['routes']))}


def classify(us, work, calls=1):
    if work is None or us <= 0:
        return 'unexplained'
    ratio = work['work_us'] / us
    if ratio > 1.15:
        return 'unexplained'  # model/profile disagreement, not super-peak efficiency
    if work['limiter'] == 'compute':
        return 'compute-bound' if ratio >= 0.5 else 'unexplained'
    if ratio >= 0.70:
        return 'memory-bound near SOL'
    if work['work_us'] <= 3.0 * max(calls or 1, 1):
        return 'latency-or-launch bound'
    return 'unexplained'


def read_profile(receipt):
    """The canonical receipt contains the same profile records as okc's duplicate files."""
    paths = list((ROOT / 'runs' / receipt['board']).glob('*_' + receipt['id'] + '.json'))
    if len(paths) != 1:
        raise ValueError(f"{receipt['id']}: expected exactly one saved receipt")
    records = receipt['result'].get('records', [])
    profiles = [record for record in records if record.get('event') == 'profile']
    if not profiles:
        return None, None
    return paths[0], dict(profiles=profiles)


def receipts(problem):
    out = []
    for p in sorted((ROOT/'runs'/problem).glob('*.json')):
        j = json.loads(p.read_text())
        r = j.get('result') or {}
        records = [v for v in r.get('records', []) if v.get('event') == 'benchmark']
        if r.get('passed') is True and r.get('status') == 'ok' and r.get('speedup') and not r.get('flags') and len(records) == len(spec(problem)['benchmarks']):
            out.append((p, j))
    return out


def source_path(j):
    for line in reversed((ROOT / 'runs/ledger.jsonl').read_text().splitlines()):
        entry = json.loads(line)
        if entry.get('job') == j['id']:
            return entry['file']
    raise ValueError(f"{j['id']}: receipt has no ledger source identity")


def choose(problem):
    rs = receipts(problem)
    cal = [(p, j) for p, j in rs if '/calibration/' in source_path(j)]
    ours = [(p, j) for p, j in rs if '/calibration/' not in source_path(j)]
    if not ours or not cal:
        raise ValueError(f'{problem}: missing complete passing ours or calibration receipt')
    return max(ours, key=lambda v: v[1]['result']['speedup']), cal[-1]


def one_run(problem, path, j, serving):
    pp, pf = read_profile(j)
    if pf is None:
        raise ValueError(f'{path}: profile unavailable')
    profiles = {case_key(v['case']): v for v in pf['profiles']}
    benchmarks = {case_key(v['case']): v for v in j['result']['records'] if v.get('event') == 'benchmark'}
    loads = {}
    for b, floor in zip(spec(problem)['benchmarks'], task_floors(problem)):
        load = b['load']
        if serving and load.startswith('decode'):
            continue
        key = case_key(b)
        record = benchmarks[key]
        pr = parse_profile(profiles[key]['report'])
        weight = b['weight']
        bounds = shape_bounds(problem, b, design_kernels(problem, 'multi'), floor)
        cw = component_work(problem, b, bounds)
        v = loads.setdefault(load, dict(judge_us=0.0, paired_baseline_us=0.0, profile_us=0.0, busy_us=0.0,
                                       practical_us=0.0, expected_us=0.0, rows={}, components={}, shapes=[]))
        v['judge_us'] += weight * record['median_ns'] / 1000
        v['paired_baseline_us'] += weight * record['baseline_ns'] / 1000
        v['profile_us'] += weight * pr['total_us']
        v['busy_us'] += weight * pr['busy_us']
        v['practical_us'] += weight * bounds['practical']
        v['expected_us'] += weight * bounds['expected']
        v['shapes'].append(dict(case=b, judge_us=record['median_ns']/1000, profile_us=pr['total_us'],
                                bounds=bounds, component_work=cw, profile_rows=pr['rows']))
        cg = collections.Counter(group(r['name']) for r in pr['rows'])
        for g, work in cw.items():
            comp = v['components'].setdefault(g, dict(us=0.0, calls=0, bytes=0.0, flops=0.0, memory_us=0.0, compute_us=0.0, work_us=0.0))
            for field in ('bytes', 'flops', 'memory_us', 'compute_us', 'work_us'):
                comp[field] += weight * work[field]
        for r in pr['rows']:
            name, g = r['name'], group(r['name'])
            # Projection reports contain one operation, whose split-K reduction is part of dense.
            if problem == 'muse_bf16_proj' and g not in ('dense', 'unattributed'):
                g = 'dense'
            comp = v['components'].setdefault(g, dict(us=0.0, calls=0, bytes=0.0, flops=0.0, memory_us=0.0, compute_us=0.0, work_us=0.0))
            comp['us'] += weight * r['us']
            comp['calls'] += weight * (r['calls'] or 0)
            # Distinguish operation identities when the same truncated name covers different work.
            identity = f"{b['layer']}: {name}" if problem == 'muse_bf16_proj' else name
            row = v['rows'].setdefault(identity, dict(name=identity, component=g, us=0.0, calls=0, bound_us=0.0,
                                                    bytes=0.0, flops=0.0, memory_us=0.0, compute_us=0.0, bounded=True))
            row['us'] += weight * r['us']
            row['calls'] += weight * (r['calls'] or 0)
            # Main FA3 and split-K reductions don't get duplicated total attention/dense bounds.
            identifiable = cg[g] == 1 and g in cw and not name.startswith('[omitted:')
            if 'moe_gemm_kernel<0,' in name or 'moe_gemm_kernel<1,' in name:
                # Prefill MODE 1 is gate/up; MODE 0 is down (source template).
                identifiable = True
                fraction = 2/3 if '<1,' in name else 1/3
            else:
                fraction = 1.0
            if identifiable:
                work = cw[g]
                row['bound_us'] += weight * fraction * work['work_us']
                for field in ('bytes', 'flops', 'memory_us', 'compute_us'):
                    row[field] += weight * fraction * work[field]
            else:
                row['bounded'] = False
    for v in loads.values():
        for row in v['rows'].values():
            row['share_pct'] = 100 * row['us'] / v['profile_us']
            work = dict(work_us=row['bound_us'], memory_us=row['memory_us'], compute_us=row['compute_us'],
                        limiter='compute' if row['compute_us'] > row['memory_us'] else 'memory') if row['bounded'] else None
            row['bound_pct'] = 100 * row['bound_us']/row['us'] if row['bounded'] and row['us'] else None
            row['class'] = classify(row['us'], work, row['calls'])
        for comp in v['components'].values():
            comp['limiter'] = 'compute' if comp['compute_us'] > comp['memory_us'] else 'memory'
            comp['bound_pct'] = 100 * comp['work_us']/comp['us'] if comp['us'] and comp['work_us'] else None
            comp['class'] = classify(comp['us'], comp if comp['work_us'] else None, comp['calls'])
    return dict(receipt=str(path.relative_to(ROOT)), job=j['id'], source=source_path(j),
                profile=str(pp.relative_to(ROOT)), speedup=j['result']['speedup'],
                ci=j['result'].get('speedup_ci'), loads=loads)


def build(problem, serving=False):
    ours, stock = choose(problem)
    return dict(problem=problem, ours=one_run(problem, *ours, serving),
                sglang=one_run(problem, *stock, serving))


def fnum(x):
    return 'n/a' if x is None else f'{x:,.1f}'


def markdown(data):
    print('# H200 profile comparison\n')
    print('Measured H200 judge medians and GPU-profile sums are separate. Load weights are applied before summing; all times are us per step/load, not us per single call. Positive delta means ours is slower.\n')
    print('Modeled bounds: sim bytes / 4.32 TB/s or BF16 FLOPs / 741.75 TFLOP/s; max of the two. Per-kernel bound excludes launch cost. Total practical uses sim design launch costs. Efficiency = 100 × modeled work bound / measured profile time. Classes are roofline hypotheses, not hardware-counter diagnoses.\n')
    print('n/a: truncated names, a split/reduction with unknown work allocation, or unidentified glue. No guessed attribution. Component bounds are shared once across their rows; use them to compare fused vs split implementations. Prefill attention adds causal within-chunk keys omitted by sim.layer_work; total practical remains the unmodified sim result. Profiles may overlap under PDL: kernel sums are not critical-path duration.\n')
    for d in data:
        print(f"## {d['problem']}\n")
        for side in ('ours', 'sglang'):
            r=d[side]
            print(f"{side}: `{r['source']}`; [{r['job']}]({ROOT / r['receipt']}), profile [{r['profile']}]({ROOT / r['profile']}); board geomean {r['speedup']:.4f}x.\n")
        print('| Load | Ours judge H200 | Paired stock H200 | Calibration judge H200 | Ours profile sum H200 | Stock profile sum H200 | Ours / stock busy H200 | Practical total MODEL | Expected MODEL |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for load,o in d['ours']['loads'].items():
            s=d['sglang']['loads'][load]
            print(f"| {load} | {o['judge_us']:,.1f} | {o['paired_baseline_us']:,.1f} | {s['judge_us']:,.1f} | {o['profile_us']:,.1f} | {s['profile_us']:,.1f} | {o['busy_us']:,.1f} / {s['busy_us']:,.1f} | {o['practical_us']:,.1f} | {o['expected_us']:,.1f} |")
        print()
        for load,o in d['ours']['loads'].items():
            s=d['sglang']['loads'][load]
            print(f'### {load}: per-kernel diff (measured H200; bounds modeled)\n')
            print('| Kernel (saved, possibly truncated name) | Ours us / share | Stock us / share | Delta us | Ours / stock bound us | Ours / stock % of bound | Ours / stock class |')
            print('|---|---:|---:|---:|---:|---:|---|')
            for name in sorted(set(o['rows'])|set(s['rows']), key=lambda n:max(o['rows'].get(n,{}).get('us',0),s['rows'].get(n,{}).get('us',0)),reverse=True):
                a,b=o['rows'].get(name),s['rows'].get(name)
                def time(r): return f"{r['us']:,.1f} / {r['share_pct']:.1f}%" if r else '—'
                def bound(r): return fnum(r['bound_us']) if r and r['bounded'] else 'n/a'
                def pct(r): return fnum(r['bound_pct'])+'%' if r and r['bound_pct'] is not None else 'n/a'
                print(f"| {name.replace('|','/')} | {time(a)} | {time(b)} | {(a['us'] if a else 0)-(b['us'] if b else 0):+,.1f} | {bound(a)} / {bound(b)} | {pct(a)} / {pct(b)} | {a['class'] if a else '—'} / {b['class'] if b else '—'} |")
            print('\nComponent work (MODEL) versus GPU-profile time (H200); components cover only identifiable rows:\n')
            print('| Component | Ours / stock us H200 | Bytes / FLOPs MODEL | Work bound us MODEL | Ours / stock bound % | Ours / stock class |')
            print('|---|---:|---:|---:|---:|---|')
            for g in sorted(set(o['components'])|set(s['components'])):
                a,b=o['components'].get(g,{}),s['components'].get(g,{})
                w=a if a.get('work_us') else b
                amount = f"{w['bytes']:.3g} / {w['flops']:.3g}" if w.get('work_us') else 'n/a'
                work_time = fnum(w['work_us']) if w.get('work_us') else 'n/a'
                print(f"| {g} | {a.get('us',0):,.1f} / {b.get('us',0):,.1f} | {amount} | {work_time} | {fnum(a.get('bound_pct'))} / {fnum(b.get('bound_pct'))} | {a.get('class','—')} / {b.get('class','—')} |")
            print()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('problem', nargs='?', default='all', choices=('all',)+PROBLEMS)
    ap.add_argument('--serving', action='store_true', help='only verify loads and prefill')
    ap.add_argument('--json', action='store_true', help='machine-readable attribution and model inputs')
    args=ap.parse_args()
    data=[build(p,args.serving) for p in (PROBLEMS if args.problem=='all' else (args.problem,))]
    if args.json:
        print(json.dumps(data,indent=2))
    else:
        markdown(data)


if __name__=='__main__':
    main()
