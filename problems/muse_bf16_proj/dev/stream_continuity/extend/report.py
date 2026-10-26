"""Paired six-load G0 inference; seed, not replay, is the independent unit."""
import argparse,ast,base64,hashlib,json,lzma,math,statistics,struct
from pathlib import Path
H=Path(__file__).resolve().parent
T95=2.131449545559323
BOUND=1.0855676

def ci(logs):
    m=statistics.mean(logs);se=statistics.stdev(logs)/math.sqrt(len(logs))
    return dict(R=math.exp(m),lo=math.exp(m-T95*se),hi=math.exp(m+T95*se),log_se=se)

def decode(path):
    if path.suffix=='.xz':return json.loads(lzma.decompress(path.read_bytes()))
    text=path.read_text()
    if path.suffix=='.json':
        obj=json.loads(text)
        import sys
        sys.path.insert(0,str(H.parents[4]/'tools'))
        from lab_packet import result_lines
        text='\n'.join(result_lines(obj['result']['log_tail']))
    import re
    blob=re.search(r'EW_XZ85 ([!-~]+)',text).group(1)
    # JSON receipts escape quotation/backslash; the base85 alphabet uses neither.
    raw=lzma.decompress(base64.b85decode(blob));size=struct.unpack_from('<I',raw)[0]
    x=json.loads(raw[4:4+size]);binary=raw[4+size:]
    x['_sample_values']=struct.unpack('<'+'i'*(len(binary)//4),binary)
    return x

def analyze(x):
    names=x['sample_arms'];loads=x['sample_loads'];n=len(x['seeds'])
    failures={c['case']:c['failures'] for c in x['cases'] if c['failures']}
    assert n==16 and x['TASK_checks']==3456 and len(x['cases'])==12
    vals=x['_sample_values'];assert len(vals)==len(loads)*n*len(names)
    paired={a:[[math.log(1+vals[(l*n+i)*len(names)+j]/x['sample_scale']) for l in range(6)] for i in range(n)] for j,a in enumerate(names)}
    out={}
    for arm,v in paired.items():
        row=ci([statistics.mean(s) for s in v]);row['TASK_failures']={case:bad[arm] for case,bad in failures.items() if arm in bad}
        row['G0']=not row['TASK_failures'] and row['lo']>1 and arm not in ('A2','G')
        guard=2*x.get('ratio_max_rounding_error',0)/(min(math.exp(z) for s in v for z in s)-x.get('ratio_max_rounding_error',0))
        row['G0_lower_with_rounding']=row['lo']*math.exp(-guard)
        row['G0']=row['G0'] and row['G0_lower_with_rounding']>1
        row['paired_seed_R']=[math.exp(statistics.mean(s)) for s in v]
        row['loads']={load:ci([s[l] for s in v]) for l,load in enumerate(loads)}
        row['bound_log_share']=math.log(row['R'])/math.log(BOUND)
        out[arm]=row
    for early,late in (('ALL','LATE'),('KV','KV_LATE'),('H2','G'),('B1','G')):
        out[early+'_vs_'+late]=ci([statistics.mean(a-b for a,b in zip(v,w)) for v,w in zip(paired[early],paired[late])])
    return dict(base_sha256=x['base_sha256'],TASK_checks=x['TASK_checks'],failures=failures,seeds=x['seeds'],reps=x['reps'],arms=out,resources=x.get('resources'),profiles={c['case']:c['proof'] for c in x['cases']},elapsed_s=x['elapsed_s'],CI_method='Student t95 on16 paired seed log six-load ratios;df15. G0 additionally guards0.5ppm ratio rounding; no AA correction or per-load veto.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    result=analyze(decode(a.receipt));result.update(receipt=str(a.receipt),receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest());s=json.dumps(result,indent=2)+'\n'
    if a.receipt.suffix=='.json':
        tree=ast.parse(a.receipt.with_suffix('.py').read_text())
        entries=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_LAB_PAYLOADS')
        src=next(e[2] for e in entries if e[0]=='lab_owned')
        manifest=json.loads((H/'lab_manifest.json').read_text());sha=hashlib.sha256(src.encode()).hexdigest()
        assert sha==manifest['payload_sha256'],'receipt source differs from frozen manifest'
        result.update(payload_sha256=sha,arm_sha256=manifest['arm_sha256'])
        s=json.dumps(result,indent=2)+'\n'
    if a.out:a.out.write_text(s)
    for arm,r in result['arms'].items():print(f"{arm:12s} {r['R']:.6f} [{r['lo']:.6f}, {r['hi']:.6f}] {'G0' if r.get('G0') else ''}")
    print('TASK',result['TASK_checks'],'failures',result['failures'])
