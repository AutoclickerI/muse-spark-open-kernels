"""Direct named-kernel profile accounting; never infer hidden kernel times."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[4]
JOBS=('a30fb411b0c74ffb','99820fa6e5c14fe1')
NAMES=('_boundary_kernel','_canonical_expand_coalesced','_portfolio_swiglu','_swiglu_kernel',
       '_combine_kernel','_gather_norm_kernel','_routing_kernel','_pre_expert_norm_kernel',
       '_norm_gate_kernel','_qk_prep_kernel','_offsets_kernel','_gather_cache_kernel')


def byte_bounds(name,case):
    t=case['batch']*case['q_len'];h=8192;d=4096
    if name=='_boundary_kernel':return 24*t*h,24*t*h # both boundaries:12B/element each
    if name=='_canonical_expand_coalesced':
        if t<8192:return 0,0 # Humming consumes packed NVFP4 directly.
        one=3*d*d*(.5+1/16+2) # active expert's packed reads + BF16 writes
        return one,32*one
    if name in ('_portfolio_swiglu','_swiglu_kernel'):
        if (name=='_portfolio_swiglu')!=(t>=8192):return 0,0
        return 6*t*d,6*8*t*d
    if name in ('_combine_kernel','_gather_norm_kernel'):return 4*t*d,18*t*d
    if name=='_routing_kernel':return t*(512*4+8*4*4)+8*t,t*(512*4+8*4*4)+64*t
    if name=='_pre_expert_norm_kernel':return 0,0 # Already part of gather_norm in these winners.
    if name=='_norm_gate_kernel':return 6*t*1024,6*t*1024
    if name=='_qk_prep_kernel':return t*(3*1280+16+256),t*(3*1280+16+256)
    if name=='_offsets_kernel':return 32*12,32*12
    if name=='_gather_cache_kernel':
        b=case['batch']*(case['ctx']+case['q_len'])*512
        return b,b # selected fa3_contiguous uses FP8 output
    raise KeyError(name)


def main():
    result=[]
    for job in JOBS:
        p=next((ROOT/'problems/muse_prefill_layer/profiles').glob('*'+job+'.json'))
        j=json.loads(p.read_text())
        for load in ('turn 1','turn 4','first turn'):
            profs=[r for r in j['profiles'] if r['case']['load']==load]
            weight=sum(p['case']['weight'] for p in profs)
            for name in NAMES:
                times=[];lo=hi=0
                for p in profs:
                    case=p['case'];w=case['weight']
                    match=re.search(r'^'+re.escape(name)+r'\s+([\d.]+) µs',p['report'],re.M)
                    times.append(float(match[1]) if match else None)
                    b0,b1=byte_bounds(name,case);lo+=w*b0/4.8e6;hi+=w*b1/4.8e6
                # A missing top-12 entry is unknown, not zero.
                measured=(sum(t*p['case']['weight'] for t,p in zip(times,profs))/weight
                          if all(t is not None for t in times) else None)
                result.append(dict(job=job,load=load,kernel=name,mean_us=measured,
                    visible_us=times,floor_us=[lo/weight,hi/weight],
                    weighted_ms=None if measured is None else measured*weight/1000))
    out=ROOT/'runs/screen/glue_prefill/inventory.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Prefill glue: direct H200 inventory','',
        'Named-kernel times from board-best a30fb411b0c74ffb and99820fa6e5c14fe1.',
        'Each column is the official multiplicity-weighted mean us per layer call for that load.',
        'Multiply by62/62/186 to get full-load us. Reports contain only12 kernels: absent is N/A.',
        'Byte floors use4.8TB/s and compulsory activation read/write traffic; cached vectors/metadata',
        'are omitted where small. They do not model SFU/reduction/atomic work or launch latency.',
        'For expert-dependent rows, R local pairs is bounded by T..8T; touched experts E by1..32.',
        'Those ranges are bounds, not an inferred routing count. Separate dequant is counted once.',
        'Router uses sigmoid plus top8 weight renormalization, not softmax.','',
        '| Kernel | turn1 us | floor | turn4 us | floor | first-turn us | floor |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        vals=[]
        for load in ('turn 1','turn 4','first turn'):
            rs=[r for r in result if r['kernel']==name and r['load']==load]
            ts=[r['mean_us'] for r in rs if r['mean_us'] is not None]
            us='N/A' if not ts else f'{min(ts):.2f}–{max(ts):.2f}'
            lo,hi=rs[0]['floor_us'];floor=f'{lo:.2f}' if abs(lo-hi)<1e-8 else f'{lo:.2f}–{hi:.2f}'
            vals.extend((us,floor))
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines+=['','The two boundaries move24*T*8192 bytes combined: at16K,3.221GB and671.09us.',
        'A single16Kx8192 BF16 read OR write is268.44MB /55.92us; a read+write is111.85us.',
        'Boundary time is about159ms per first turn, with a125ms traffic floor (~34ms ideal headroom).',
        'The earlier235ms remainder comes from a different run/stage accounting; this inventory uses',
        'the named kernels in the current best and does not subtract separately-counted dequant twice.',
        'Detailed per-receipt/per-profile values: runs/screen/glue_prefill/inventory.json.','']
    (ROOT/'docs/reference/prefill_glue.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':main()
