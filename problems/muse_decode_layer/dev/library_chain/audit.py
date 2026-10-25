"""Source dispatch census and fixture-specific CUPTI boundary estimates."""
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
import build
H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    s=build.BASE.read_text();assert hashlib.sha256(s.encode()).hexdigest()==build.SHA
    receipt=next((R/'runs').glob('*/*c6bd22e2084c4809.json'))
    tail=json.loads(receipt.read_text())['result']['log_tail']
    packed=next(l[8:] for l in tail.splitlines() if l.startswith('TL_XZ85 '))
    tl=json.loads(lzma.decompress(base64.b85decode(packed)))
    timeline=[]
    for case in tl['results']:
        for i,node in enumerate(case['nodes']):
            name=tl['names'][node[0]];stage=tl['stages'][node[1]]
            if 'nvjet' not in name and 'flash::' not in name and name!='_attn_sumcast_kernel':continue
            nxt=case['nodes'][i+1] if i+1<len(case['nodes']) else None
            timeline.append(dict(case=case['case'],stage=stage,name=name,start_us=node[4],end_us=node[5],
                residence_us=round(node[5]-node[4],6),next_start_gap_us=round(nxt[4]-node[5],6) if nxt else None))
    # An explicit overlap budget, NOT a measured producer-tail decomposition.
    # One stream intercept per external input/output edge is a useful screening
    # allowance. Never count all producer residence as removable PDL overhead.
    allowance=2.36
    rows=[]
    for load,b,l in [('d1',1,1),('d16',16,1),('d32',32,1),('v1',1,8),('v16',16,8),('v32',32,8)]:
        if l==1:
            if b>1:
                gaps=[max(0.,x['next_start_gap_us']) for x in timeline if x['case'] in (load+'S',load+'F') and x['name']=='_attn_sumcast_kernel']
                rows.append(dict(load=load,call='_attn_sumcast (owned)',shape=f'S x {b} x 1280/2304 -> BF16',
                    kernel_count_per_layer=1,layers=62,kernels_per_step=62,
                    external=False,gap_us=gaps,model='owned release already at entry; ordinary attn_main consumer; S15e tests its enable/wait edge'))
            else:
                rows.append(dict(load=load,call='none',shape='',kernel_count_per_layer=0,layers=62,kernels_per_step=0,external=False))
            continue
        # FA3 has one metadata scheduler, one main and one combine in c6bd22e2.
        # Its own internal nodes overlap; three kernels are NOT three ordinary
        # completion boundaries. Count the two outer edges as one call group.
        layers=62 if b==1 else 46
        shapes=['S','F'] if b==1 else ['S']
        estimates=[]
        for sf in shapes:
            c=load+sf;nn=[x for x in timeline if x['case']==c and x['stage']=='attn' and 'flash::' in x['name']]
            original=next(r for r in tl['results'] if r['case']==c)
            prep=next(n for n in original['nodes'] if tl['names'][n[0]]=='_qk_prep_kernel')
            gap=max(0.,nn[0]['start_us']-prep[5])+max(0.,nn[-1]['next_start_gap_us'])
            estimates.append(dict(layer=sf,gap_us=round(gap,3),gap_plus_tail_allowance_us=round(gap+2*allowance,3),
                                  library_residence_span_us=round(nn[-1]['end_us']-nn[0]['start_us'],3)))
        total=sum((46 if e['layer']=='S' else 16)*e['gap_plus_tail_allowance_us'] for e in estimates)
        rows.append(dict(load=load,call='FA3 metadata + main + combine',shape=f'Q[{b},8,16,64], FP8 KV, page1, ctx~80K',
                         kernel_count_per_layer=3,layers=layers,kernels_per_step=3*layers,external=True,
                         estimates=estimates,modeled_step_us=round(total,3)))
        if b==1:continue
        for role,n,k in [('qkv_gate',2304,8192),('o',8192,1024),('pool_router',4608,8192),('unpooler',8192,4096)]:
            if b==16 and role=='qkv_gate':continue  # Owned VDSK, no cast/reduce launch.
            estimates=[]
            for sf in ('S','F'):
                item=next(x for x in timeline if x['case']==load+sf and x['stage']==role and 'nvjet' in x['name'])
                gap=max(0.,item['next_start_gap_us'])
                estimates.append(dict(layer=sf,gap_us=round(gap,3),gap_plus_tail_allowance_us=round(gap+allowance,3),
                                      library_residence_span_us=item['residence_us']))
            rows.append(dict(load=load,call=role+' cuBLAS/Lt',shape=f'({b*8},{n},{k}) M,N,K',kernel_count_per_layer=1,
                layers=62,kernels_per_step=62,external=True,estimates=estimates,
                modeled_step_us=round(46*estimates[0]['gap_plus_tail_allowance_us']+16*estimates[1]['gap_plus_tail_allowance_us'],3)))
    out=dict(base_sha256=build.SHA,rows=rows,timeline=timeline,source_receipt=str(receipt.relative_to(R)),
        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),producer_tail_allowance_per_edge_us=allowance,
        caveats=['S11 fixture-specific activity, not an S15 timing claim',
                 'tail is not identifiable from kernel timestamps; gap+2.36us/edge is a screening model, not a gain claim',
                 'FA3 metadata/main/combine overlap internally in the receipt; external is not proof of implicit-exit scheduling',
                 'Lt algorithm may be a multi-kernel implementation; listed count is observed c6bd22e2, source plans do not guarantee it',
                 'zero separate torch BF16 cast on scored Hopper paths; _attn_port_raw .to is a dtype-preserving alias after owned sumcast',
                 'prepare-only repacks, scale casts, counter initialization and inverse-frequency cache creation are outside captured steady-state kernel_fn',
                 'no standalone q/k FP8 cast: fused into owned qk_prep, or VQ8/attn_main'])
    (H/'boundary_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in rows:print(r['load'],r['call'],r['kernels_per_step'],r.get('modeled_step_us',''))


if __name__=='__main__':main()
