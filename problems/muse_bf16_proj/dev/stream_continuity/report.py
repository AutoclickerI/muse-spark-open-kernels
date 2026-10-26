"""Decode lossless paired seed archive and apply GOAL G0 without per-load vetoes."""
import argparse,base64,hashlib,json,lzma,math,statistics,struct
from pathlib import Path
H=Path(__file__).resolve().parent
LOADS=('d1','d16','d32','v1','v16','v32')

def decode(blob):
    raw=lzma.decompress(base64.b85decode(blob));size=struct.unpack_from('<I',raw)[0];x=json.loads(raw[4:4+size]);binary=[];v=shift=0
    for c in raw[4+size:]:
        v|=(c&127)<<shift
        if c&128:shift+=7
        else:binary.append((v>>1)^-(v&1));v=shift=0
    assert not shift
    arms=x['arms'];offset=0
    for c in x['cases']:
        c['samples']=[]
        for _ in range(c['sample_rows']):
            c['samples'].append({a:(c['means_ns'][a]+binary[offset+i])/1000 for i,a in enumerate(arms)});offset+=len(arms)
    assert offset==len(binary)
    return x

def interval(logs):
    # Student t95 for the frozen n=16 paired seed cohort, on log R.
    assert len(logs)==16
    mean=statistics.mean(logs);se=statistics.stdev(logs)/math.sqrt(len(logs));half=2.131449545559323*se
    return dict(R=math.exp(mean),CI95=[math.exp(mean-half),math.exp(mean+half)],log_SE=se,n=len(logs))

def evaluate(x):
    cases={c['case']:c for c in x['cases']};assert set(cases)=={l+w for l in LOADS for w in ('S','F')}
    failures={k:c['failures'] for k,c in cases.items() if c['failures']}
    assert x['task_checks']==12*len(x['arms'])*len(x['seeds'])
    by_arm={}
    for arm in x['arms'][1:]:
        per_load={};per_seed=[];shape={}
        for load in LOADS:
            logs=[]
            for i,seed in enumerate(x['seeds']):
                old=sum(w*cases[load+k]['samples'][i]['A'] for k,w in (('S',46),('F',16)))
                new=sum(w*cases[load+k]['samples'][i][arm] for k,w in (('S',46),('F',16)))
                logs.append(math.log(old/new))
            per_load[load]=interval(logs)
        for i,seed in enumerate(x['seeds']):
            z=[]
            for load in LOADS:
                old=sum(w*cases[load+k]['samples'][i]['A'] for k,w in (('S',46),('F',16)))
                new=sum(w*cases[load+k]['samples'][i][arm] for k,w in (('S',46),('F',16)))
                z.append(math.log(old/new))
            per_seed.append(dict(seed=seed,R=math.exp(statistics.mean(z))))
        for k,c in cases.items():shape[k]=interval([math.log(row['A']/row[arm]) for row in c['samples']])
        r=interval([math.log(s['R']) for s in per_seed]);r.update(per_load=per_load,per_shape=shape,per_seed=per_seed)
        r['passes_G0']=not failures and r['CI95'][0]>1 and arm!='A2';by_arm[arm]=r
    return dict(base_sha256=x['base_sha256'],seeds=x['seeds'],reps=x['reps'],task_checks=x['task_checks'],failures=failures,arms=by_arm,preselected=x['preselected'],CI_method='Two-sided Student t interval on log six-load R across16 paired held-out seeds; df15; no AA correction or per-load veto',profiles={k:c['profiles'] for k,c in cases.items()},banks={k:c['banks'] for k,c in cases.items()})

def main():
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();r=json.loads(a.receipt.read_text())
    lines=r['result']['log_tail'].splitlines();blobs=[l.split('CONTINUITY_XZ85 ',1)[1] for l in lines if 'CONTINUITY_XZ85 ' in l];assert len(blobs)==1,lines[-10:]
    x=decode(blobs[0]);report=evaluate(x);report.update(receipt=str(a.receipt),receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest())
    (H/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    for arm,row in report['arms'].items():print(arm,f"R={row['R']:.7f}",f"CI95=[{row['CI95'][0]:.7f},{row['CI95'][1]:.7f}]",'PASS_G0' if row['passes_G0'] else '')
    print('TASK',report['task_checks'],'failures',report['failures'])
if __name__=='__main__':main()
