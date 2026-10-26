"""Classify only complete, gated H200 receipts; a failed arm is unmeasured."""
from pathlib import Path
import json,re,datetime
H=Path(__file__).resolve().parent;R=H.parents[3]
ledger=[json.loads(l) for l in (R/'runs/ledger.jsonl').read_text().rsplit('\n',1)[0].splitlines()]
def analyse(sha,kind):
    matches=[x for x in ledger if x['sha256']==sha]
    if not matches:return dict(kind=kind,status='pending')
    entry=matches[-1];file=R/Path(entry['source']).with_suffix('.json');d=json.loads(file.read_text());r=d.get('result') or {};tail=r.get('log_tail','')
    name='p10_sparse_pool_lab' if kind=='pool' else 'p10_all_roles_lab'
    found=re.search(r'^LAB_BEGIN '+name+r'\s+.*?\n(.*?)(?:^LAB_END.*?$|\Z)',tail,re.M|re.S)
    body=found[1] if found else tail;rows=[]
    if kind=='pool':
        rx=r'^R t=(\d+) (\S+) A=([\d.]+) B=([\d.]+) r=([\d.]+)\[([\d.]+),([\d.]+)\] aa=([\d.]+) e=([\deE+.-]+) p=([\deE+.-]+)'
    else:
        rx=r'^R (\d+):(\S+):([BC]) ([\d.]+) ([\d.]+) r=([\d.]+)\[([\d.]+),([\d.]+)\] aa=([\d.]+) e=([\deE+.-]+) p=([\deE+.-]+)'
    for match in re.finditer(rx,body,re.M):
        a=match.groups()
        if kind=='pool':t,arm=a[:2];role='pool_router';arm=arm;values=a[2:]
        else:t,role,arm=a[:3];values=a[3:]
        av,bv,ratio,lo,hi,aa,error,probe=map(float,values)
        # Exactly the all-role README rule; pool uses the same conservative
        # envelope for selecting whole-layer follow-up candidates.
        margin=max(.02,2*aa);eligible=ratio<1-margin and hi<=1+aa
        rows.append(dict(t=int(t),role=role,arm=arm,A_us=av,B_us=bv,ratio=ratio,ratio_min=lo,ratio_max=hi,aa_max=aa,error=error,probe=probe,
            followup_candidate=eligible,classification='whole-layer candidate' if eligible else 'loss' if lo>1+aa else 'unresolved within control spread'))
    expected=48 if kind=='pool' else 74
    complete=len(rows)==expected and ('END elapsed=' in body if kind=='pool' else 'END complete=57/57 direct=17' in body)
    valid=complete and not any(s in body for s in ('LAB_ERROR','INCOMPLETE','LAB_TRUNCATED','AssertionError','Traceback'))
    if not valid:
        for x in rows:x['followup_candidate']=False;x['classification']='unmeasured: incomplete or failed gate'
    return dict(kind=kind,job=entry['job'],status='measured' if valid else 'unmeasured',complete=complete,expected_rows=expected,
        rows_found=len(rows),host_passed=bool(r.get('passed')),host_error=r.get('error'),receipt=str(file.relative_to(R)),rows=rows,
        error_lines=[s for s in body.splitlines() if any(t in s for t in ('LAB_ERROR','INCOMPLETE','TRUNCATED','Error','assert '))],
        caveat='lab validity uses its own gates; the host board score is not the lab result')
out=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=[
    analyse('1a03dfa6906c8242f25c8d808aec8ef96ab6de960f905257195369842bf021dd','pool'),
    analyse('d50be2e7d31519d68877a08613c11ab0bdac83caa74086708ca177382dae914b','all')])
(H/'results.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out['runs']:
    print(r['kind'],r['status'],r.get('job'),r.get('rows_found'))
    for row in r.get('rows',[]):
        if row['followup_candidate']:print('P4_CANDIDATE',row)
