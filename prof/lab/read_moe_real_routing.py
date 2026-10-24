#!/usr/bin/env python3
"""Parse lab receipts into a complete kernel x M x routing table and dispatch evidence."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MS=(8,16,32,128,256,512,753)
CHOICES=tuple((w,s,f,n) for w,s in ((32,3),(64,3),(64,5),(32,5)) for f,n in ((1,8),(1,16),(.5,8)))


def read(ref):
    p=Path(ref)
    if not p.exists():
        hits=sorted(ROOT.glob(f'runs/*/*_{ref}.json'))
        if not hits:raise ValueError(f'no receipt for {ref}')
        p=hits[-1]
    p=p.resolve()
    receipt=json.loads(p.read_text());log=receipt['result'].get('log_tail','')
    if 'LAB_BEGIN' not in log:raise ValueError(f'no lab in {p}')
    body=log[log.index('LAB_BEGIN'):]
    if 'LAB_END' not in body:raise ValueError(f'incomplete lab log in {p}')
    source=p.with_suffix('.py').read_text() if p.with_suffix('.py').exists() else ''
    remote_ref='dst=lab_torch.zeros((33,n,4096)' in source
    rows=[]
    for line in body.splitlines():
        m=re.match(r'T (\S+) (\d+) ([RB]) ([\d.]+)(?: p=([^ ]+))?',line)
        if m:
            phase_error=re.search(r'\bpERR=(\S+)',line)
            rows.append(dict(kernel='bf16ref_remote' if m[1]=='bf16ref' and remote_ref else m[1],m=int(m[2]),routing=m[3],us=float(m[4]),phases=m[5],phase_error=phase_error[1] if phase_error else None,job=receipt.get('job') or receipt.get('job_id') or p.stem.split('_')[-1]))
    captured=re.search(r'META CAPTURED_ROUTING (\w+)',body)
    return dict(receipt=str(p.relative_to(ROOT)),job=p.stem.split('_')[-1],sha256=hashlib.sha256(source.encode()).hexdigest() if source else receipt.get('content_hash'),loaded='exact_loaded_decode_weights' in body,captured_from=captured[1] if captured else None,verify_corrected='tokens // q_len' in source,bf16_remote_work=remote_ref,rows=rows,log=body[:body.index('LAB_END')]+body[body.index('LAB_END'):].splitlines()[0])


def render(records):
    out=['# MoE H200 lab measurements','',
         'Times are GPU microseconds from cold-L2 CUDA graph replay, anchored by median paired ratios against V9. The judge’s roughly 4 us fixed graph cost is included. R uses the decode-style fixture; B uses the standalone MoE board generator. `exact loaded` means the lab received the decode host’s router, bias, norms, pooler, and packed experts.','']
    for rec in sorted(records,key=lambda r:(not r['loaded'],not bool(r.get('captured_from')))):
        scope='exact loaded' if rec['loaded'] else (f"loaded route IDs from {rec['captured_from']}, fixture 0" if rec.get('captured_from') else 'generated router model / reference')
        out += [f"## {rec['job']} — {scope}",'',f"Receipt: `{rec['receipt']}`",'', '| Kernel / routing | '+' | '.join(map(str,MS))+' |','|---|'+'---:|'*len(MS)]
        rows={(r['kernel'],r['m'],r['routing']):r for r in rec['rows']}
        errors={}
        for line in rec['log'].splitlines():
            failure=re.match(r'ERR (\w+)/(\d+)/([RB]) (.*)',line)
            if failure:errors[failure[1],int(failure[2]),failure[3]]=failure[4]
        names=list(dict.fromkeys([r['kernel'] for r in rec['rows']]+[k for k,m,r in errors]))
        for k in names:
            for route in ('R','B'):
                out.append(f'| {k} / {route} | '+' | '.join(f"{rows[k,m,route]['us']:.1f}" if (k,m,route) in rows else ('ERR' if (k,m,route) in errors else '—') for m in MS)+' |')
        out += ['',f"Verify-block sharing: {'corrected for every 8-token block' if rec['verify_corrected'] else 'earlier fixture; M128/256 use decode-style sharing'}. `v9p` is the native V9_phase probe; `v11` is its wide Humming path. `humming` uses the original indexed Humming settings."]
        if rec['bf16_remote_work']:
            out += ['`bf16ref_remote` computes an additional zero-weight expert for remote routes; it is an extra-work reference and is superseded by the local-only BF16 reference.']
        out += ['','Measured minima among admissible candidates (BF16/FlashInfer reference banks excluded):','', '| M | R winner | R us | B winner | B us |','|---:|---|---:|---|---:|']
        for m in MS:
            cells=[]
            for route in ('R','B'):
                rank=sorted((r['us'],r['kernel']) for r in rec['rows'] if r['m']==m and r['routing']==route and 'ref' not in r['kernel'])
                cells += [rank[0][1],f'{rank[0][0]:.1f}'] if rank else ['—','—']
            out.append('| '+str(m)+' | '+' | '.join(cells)+' |')
        phases=[r for r in rec['rows'] if r['phases']]
        if phases:
            out += ['','Phase-only graph times (W13+SwiGLU / W2 / combine); routing is included only in the end-to-end timing. Each phase starts with cold L2, so sums need not equal the end-to-end graph.','']
            out += [f"- {r['kernel']} M={r['m']} {r['routing']}: {r['phases']} us" for r in phases]
        if any(line.startswith('CY ') for line in rec['log'].splitlines()):
            out += ['','`CY` counters are per-CTA median/max clock64 cycles from single instrumented eager launches, without an explicit prelaunch L2 flush. They diagnose phase balance; they are not paired cold-graph phase wall times.']
        err=[s for s in rec['log'].splitlines() if s.startswith(('ERR','BUDGET','LAB_ERROR','LAB_TRUNCATED','LIB','END','ROUTER','ROUTE','CY '))]
        err += [f"PHASE_ERR {r['kernel']} M={r['m']} {r['routing']} {r['phase_error']}" for r in rec['rows'] if r.get('phase_error')]
        out += ['','```text',*err,'```','']
    out += ['## TMA choice key','','| Name | max rows | stages | grid / SMs | minimum rows |','|---|---:|---:|---:|---:|']
    out += [f'| tg{i} | {w} | {s} | {f} | {n} |' for i,(w,s,f,n) in enumerate(CHOICES)]
    return '\n'.join(out)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipts',nargs='+');p.add_argument('-o',type=Path,default=ROOT/'prof/lab/moe_real_routing_h200.md');a=p.parse_args()
    records=[read(r) for r in a.receipts]
    a.o.write_text(render(records));a.o.with_suffix('.json').write_text(json.dumps(records,indent=2)+'\n')
    print(a.o,hashlib.sha256(a.o.read_bytes()).hexdigest())

if __name__=='__main__':main()
