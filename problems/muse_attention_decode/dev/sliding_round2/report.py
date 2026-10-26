"""Audit P5 receipts and nominate every measured per-shape point winner to P4."""
import argparse,base64,json,lzma,math,statistics
from pathlib import Path

def stats(values):
    return statistics.mean(values),statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else math.inf

def compare(times,name,denominator='A'):
    assert len(times[name])==len(times[denominator])>1
    return stats([x/y for x,y in zip(times[name],times[denominator])])

def rank(times,name):
    mean,se=compare(times,name);noise,noise_se=compare(times,'A2')
    aa=[x/y for x,y in zip(times['A2'],times['A'])]
    other,other_se=compare(times,name,'A2')
    controls={n:compare(times,n) for n in times if n.startswith('A') and n[1:].isdigit()}
    return dict(ratio=mean,se=se,aa=noise,aa_se=noise_se,aa_range=[min(aa),max(aa)],
                baseline_controls=controls,vs_a2=other,vs_a2_se=other_se,
                point_winner=mean<1)

def report(row,marker,invalid_control=False):
    times=row['times']
    if isinstance(times,list):times=dict(zip(row['names'],times))
    expected=row['seeds']
    assert all(len(v)==expected for v in times.values()),'missing seed samples'
    counts=row.get('task_passed',row.get('task'))
    if counts is not None:
        assert set(counts)==set(times),'missing TASK arm'
        assert all(v==expected for v in counts.values()),'incomplete TASK checks'
    elif marker=='EH_XZ85':
        counts=row['task_gates'];assert set(counts)==set(times)
        assert all(len(v)==expected for v in counts.values())
    if marker=='DS_XZ85':
        assert len(row['gates'])==expected
        assert all(all(math.isfinite(v) and v<(.005 if k=='partial' else .04)
                       for k,v in g.items()) for g in row['gates'])
        for name in times:
            if name.startswith('C'):assert row['res'][name[1:]][-1]>0
    label=row.get('case','d1S' if marker=='EH_XZ85' else 'v'+str(row.get('B',1))+'S')
    noise,noise_se=compare(times,'A2')
    groups={}
    for name,recipe in row.get('arms',{}).items():groups.setdefault(tuple(recipe),[]).append(name)
    aliases=[]
    for names in groups.values():
        for name in names[1:]:
            mean,se=compare(times,name,names[0])
            aliases.append(dict(names=[name,names[0]],ratio=mean,se=se))
    # The fixed-pool v1 cohort has an established identical-kernel control
    # failure, repaired by 6a0b3f09. Do not turn A:A or a per-load CI into a
    # new acceptance gate. Report all control estimates for P4 to inspect.
    failed_alias=invalid_control and label=='v1S'
    status=('UNMEASURED whole-layer recipes: one isolated attention gate; no replay TASK gates'
            if marker=='CS_XZ85' else 'UNMEASURED: identical-recipe control failed'
            if failed_alias else 'stage-only gates' if marker=='DS_XZ85' else 'TASK gates complete')
    print(f'{label} seeds={expected} A:A={noise:.6f} SE={noise_se:.6f}; {status}')
    result=dict(case=label,seeds=expected,controls=status,identical_recipe_controls=aliases,arms={})
    if aliases:print(' identical recipes',json.dumps(aliases))
    for name in times:
        if name=='A' or (name.startswith('A') and name[1:].isdigit()):continue
        summary=rank(times,name);measured=marker!='CS_XZ85' and not failed_alias
        summary['measured']=measured;summary['nominate_to_p4']=measured and summary['point_winner']
        if 'OLD4' in times and name!='OLD4':
            summary['vs_old'],summary['vs_old_se']=compare(times,name,'OLD4')
        if 'critical_ns' in row:
            summary['critical_delta_us']={k:(v-old)/1000 for k,v,old in
                zip(row['critical_order'],row['critical_ns'][name],row['critical_ns']['A'])}
        elif 'profiles' in row:
            inc=row['profiles'][name]['inc'];base=row['profiles']['OLD4' if marker=='EH_XZ85' else 'A']['inc']
            summary['critical_delta_us']={k:inc.get(k,0)-base.get(k,0) for k in ('qkv3','qkv_gate','attn','gate','o')}
        result['arms'][name]=summary
        print(' ',name,json.dumps(summary,sort_keys=True))
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('receipt',type=Path);ap.add_argument('--output',type=Path,help='save audited summary as JSON')
    args=ap.parse_args();receipt=json.loads(args.receipt.read_text());tail=receipt['result'].get('log_tail','')
    for line in tail.splitlines():
        if line.startswith(('LAB_ERROR','FAIL ','INCOMPLETE','LAB_TRUNCATED')):print(line)
    archives=[]
    for line in tail.splitlines():
        if line.split(' ',1)[0] in ('R2_XZ85','VR_XZ85','EH_XZ85','DS_XZ85','CS_XZ85','CG_XZ85'):
            marker,blob=line.split(' ',1);rows=json.loads(lzma.decompress(base64.b85decode(blob)))
            archives.extend((marker,r) for r in (rows if isinstance(rows,list) else [rows]))
    if not archives:
        print(tail[-4000:]);raise SystemExit('UNMEASURED: no complete P5 archive in receipt')
    result=dict(job=receipt['id'],receipt=str(args.receipt),host_passed=receipt['result'].get('passed'),
                results=[report(row,marker,receipt['id']=='6c26b14fdd28458b') for marker,row in archives])
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Every measured point ratio <1 is nominated. A:A and SE describe uncertainty. '
          'Only P4 held-out six-load weighted-score confirmation promotes a composition.')
if __name__=='__main__':main()
