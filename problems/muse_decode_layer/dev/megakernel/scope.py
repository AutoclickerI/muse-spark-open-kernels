"""Source-pinned small-M kernel accounting and six-load decision model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
RESET=ROOT/'problems/muse_decode_layer/profiles/cand_r3_best7_reset.c0e74bedf7b74be5.json'
MODEL=ROOT/'problems/muse_decode_layer/dev/levers/current.json'
BPS_US=3.6e6

def kernel_bytes(name,t,work):
    p=work['projection_bytes'];exact=True;note='one useful weight read; excludes X/output/split traffic'
    if '_gemv_kernel' in name:
        value=p['pool_router']+(p['unpooler'] if t==1 else 0)
        note='merged pool+unpool calls' if t==1 else 'pool+router'
    elif 'nvjet_sm90_tst_128x8' in name:
        value=p['qkv_gate']+(p['unpooler'] if t==8 else 0)
        note='merged QKV+unpool calls' if t==8 else 'QKV'
    elif 'nvjet_sm90_tst_64x8' in name:value=p['o'];note='O'
    elif 'moe_fused' in name:
        value=work['moe_bytes'];exact=False;note='inferred active-expert/route byte expectation from official floor; not measured traffic'
    elif 'attn_main' in name or 'enable_sm90_or_later' in name:
        value=work['attention_bytes'];note='logical KV/input bytes from bound model, not duplicate physical loads'
    elif '_boundary_fast' in name:
        value=2*t*8192*24;note='two boundaries: branch BF16,stream FP32,alpha/beta/prew FP32,new stream FP32,hidden BF16; excludes split rereads'
    elif '_route_fast' in name:
        value=t*(4608*4+256*4+4096*4+4096*2+8*12)
        note='one useful read of PR/bias/prew and output X/ids/weights/local IDs; excludes split rereads'
    elif 'splitKreduce' in name:
        value=10*t*(2304+(1024 if t==8 else 0));exact=False
        note='minimum S=2 FP32 partial loads + BF16 output; actual split count is absent from profile'
    elif '_qk_prep' in name:
        value=t*(2304*4+16*64+1024*4+2*2*64+2*8)
        note='minimum S=1 FP32 input,FP8 Q/K/V,FP32 gate,positions/cache locations; excludes inv_freq and scales'
    elif '_norm_gate' in name:
        value=t*1024*(2+4+2);note='BF16 input + FP32 gate + BF16 output'
    elif 'FlashAttnFwdCombine' in name:
        value=t*1024*(2*4+2)+t*16*2*4;exact=False
        note='minimum two FP32 partial outputs + BF16 output + partial LSE; actual split count unavailable'
    elif 'prepare_varlen_num_blocks' in name:
        value=0;exact=False;note='length/split metadata size absent from truncated profile; no bandwidth assigned'
    else:raise ValueError('unclassified profile row '+name)
    return value,exact,note

def factor(base,save):
    by={}
    for r in base:
        case=r['case'];load=case['load'];w=case['weight'];old=r['modeled_adjusted_us']
        pair=by.setdefault(load,[0.,0.]);pair[0]+=w*old
        pair[1]+=w*(old-save.get(r['index'],0.))
    return math.prod(a/b for a,b in by.values())**(1/6)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,default=RESET);args=parser.parse_args()
    profile=json.loads(args.profile.read_text());model=json.loads(MODEL.read_text())
    # current.json must be refreshed with dev/levers/build.py when S6 lands.
    if model['job']!=profile['job_id']:raise ValueError('refresh the lever model from this profile before accounting')
    model_rows={r['index']:r for r in model['rows']};rows=[];table=[]
    for prof in profile['profiles']:
        idx=prof['index'];case=prof['case'];t=case['batch']*case.get('q_len',1)
        if t>8:continue
        r=model_rows[idx];kernels=[]
        for line in prof['report'].splitlines():
            match=re.match(r'^(.*?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)\s*$',line)
            if not match:continue
            name,us,calls=match[1],float(match[2]),int(match[3]);b,exact,note=kernel_bytes(name,t,r['work'])
            kernels.append(dict(name=name,duration_us=us,calls=calls,useful_bytes=b,bytes_exact_under_model=exact,
                tbps=b/us/1e6 if b else None,stream_us=b/BPS_US if b else None,excess_us=us-b/BPS_US if b else None,byte_note=note))
        # Count each logical learned bank/KV bank once in the stage ceiling.
        # Partial/output traffic is kept in the per-kernel accounting but is
        # avoidable with fusion and cannot be added to the one-read floor.
        useful=sum(r['work']['projection_bytes'].values())+r['work']['attention_bytes']+r['work']['moe_bytes']
        total=sum(v['duration_us'] for v in kernels);floor=useful/BPS_US
        rows.append(dict(index=idx,case=case,tokens=t,kernel_count=sum(v['calls'] for v in kernels),kernels=kernels,
            profile_kernel_sum_us=total,reported_profile_total_us=r['profile_total_us'],one_read_bytes=useful,one_read_stream_us=floor,
            stage_sum_excess_us=total-floor,adjusted_layer_us=r['base_adjusted_us'],
            raw_graph_us=profile['benchmarks'][idx]['median_raw_ns']/1e3,
            raw_minus_kernel_sum_us=profile['benchmarks'][idx]['median_raw_ns']/1e3-total,
            modeled_current_us=r['modeled_adjusted_us'],critical_path_ceiling_us=r['modeled_adjusted_us']-floor))
    # Mechanism estimate: retain the current measured per-op inner loops.
    # Budget all dispatch, counters, activation transfer and synchronization
    # explicitly; do not replace those loops by copy bandwidth.
    op_savings={0:dict(boundaries=3.,route=1.5,qk_norm_gate=0.,tail_and_dispatch=3.,counter_and_resource_tax=-1.5),
                1:dict(boundaries=3.,route=1.5,qk_norm_gate=0.,tail_and_dispatch=3.,counter_and_resource_tax=-1.5),
                6:dict(boundaries=2.5,route=1.5,qk_norm_gate=2.,tail_and_dispatch=3.,counter_and_resource_tax=-2.),
                7:dict(boundaries=2.5,route=1.5,qk_norm_gate=2.,tail_and_dispatch=3.,counter_and_resource_tax=-2.)}
    savings={idx:sum(v.values()) for idx,v in op_savings.items()}
    estimate=factor(model['rows'],savings)
    # The four unsupported dimensions remain unchanged, as a small-M-only
    # design requires. Find uniform additional per-layer saving at these rows.
    lo,hi=0.,40.
    for _ in range(60):
        mid=(lo+hi)/2
        if factor(model['rows'],{i:mid for i in savings})<1.03:lo=mid
        else:hi=mid
    ceiling=factor(model['rows'],{r['index']:max(0.,r['critical_path_ceiling_us']) for r in rows})
    result=dict(profile=str(args.profile.relative_to(ROOT)),profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),profile_job=profile['job_id'],reference_tbps=3.6,
        precision='weight bytes exact; MoE inferred from floor; shared kernel names are grouped, not split by invented times',rows=rows,
        decision=dict(estimated_savings_us=savings,assumptions_us=op_savings,estimated_overall_factor=estimate,
            uniform_savings_needed_for_three_percent_us=hi,optimistic_one_read_ceiling_factor=ceiling,
            no_counter_tax_factor=factor(model['rows'],{i:sum(v for k,v in op_savings[i].items() if k!='counter_and_resource_tax') for i in savings}),
            build_first_increment=estimate>1.03,
            reason='Best existing inner loops retained; dense/MoE intrinsic inefficiency is not fusion savings. Muse has serial global norm/router dependencies, unlike Cohere parallel branches.'))
    (HERE/'scope.json').write_text(json.dumps(result,indent=2)+'\n')
    out=['| Load/layer | Calls | Kernel sum us | Stream floor us | Sum excess us | Adjusted layer us | Critical-path ceiling us |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        out.append(f'| {r["case"]["load"]}/{r["case"]["layer"]} | {r["kernel_count"]} | {r["profile_kernel_sum_us"]:.2f} | {r["one_read_stream_us"]:.2f} | {r["stage_sum_excess_us"]:.2f} | {r["modeled_current_us"]:.2f} | {r["critical_path_ceiling_us"]:.2f} |')
    for r in rows:
        out.extend(['',f'**{r["case"]["load"]}, {r["case"]["layer"]}**','',
            '| Kernel (same-name calls grouped) | Calls | us | Useful bytes | TB/s | Excess us |',
            '|---|---:|---:|---:|---:|---:|'])
        for k in r['kernels']:
            flag='' if k['bytes_exact_under_model'] else '~'
            rate=f'{k["tbps"]:.3f}' if k['tbps'] is not None else 'unknown'
            excess=f'{k["excess_us"]:.2f}' if k['excess_us'] is not None else 'unknown'
            name=k['name'].replace('|','/')
            byte_text=f'{flag}{k["useful_bytes"]:,.0f}' if k['useful_bytes'] else 'unknown'
            out.append(f'| `{name}` | {k["calls"]} | {k["duration_us"]:.2f} | {byte_text} | {rate} | {excess} |')
    out.extend(['',f'Estimated whole-board gain: {(estimate-1)*100:.3f}%; uniform saving needed per small-M row: {hi:.3f}us.'])
    (HERE/'accounting.md').write_text('\n'.join(out)+'\n')
    print(json.dumps(result['decision'],indent=2))

if __name__=='__main__':main()
