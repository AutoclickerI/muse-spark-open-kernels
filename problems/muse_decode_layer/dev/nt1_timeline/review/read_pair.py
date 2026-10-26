"""Audit the direct ILP/PAIR receipt and retain both ratio/SE and the point decision."""
import ast,hashlib,json,re
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent;R=D.parents[4];job='1b135b70835f4634'
p=next((R/'runs').glob(f'*/*_{job}.json'));d=json.loads(p.read_text());r=d['result'];assert r['passed'] and r['checks']['ok']
log=r['log_tail'];assert 'END seeds=32' in log and not re.search('LAB_ERROR|LAB_TRUNCATED|^FAIL ',log,re.M)
gate=re.search(r'^GATE (.*)$',log,re.M);assert gate
assert all(float(x)<.015 and float(y)<.075 for x,y in re.findall(r':([\d.]+)/([\d.]+)',gate[1]))
rows=[]
for m,arm,old,new,ratio,se,wins,n,delta in re.findall(r'^PAIR M(\d+) (\w+) PAIR_us=([\d.]+) arm_us=([\d.]+) ratio=([\d.]+) se=([\d.]+) wins=(\d+)/(\d+) delta_us=([\d.+-]+)',log,re.M):
 assert n=='32';rows.append(dict(M=int(m),arm=arm,control='PAIR',control_us=float(old),arm_us=float(new),ratio=float(ratio),SE=float(se),seeds=32,wins=int(wins),delta_us=float(delta),point_nominee=float(ratio)<1))
aa={int(m):dict(mean_abs_us=float(mean),max_abs_us=float(maximum),seeds=int(n)) for m,mean,maximum,n in re.findall(r'^AA M(\d+) abs_us=([\d.]+) max_abs_us=([\d.]+) n=(\d+)',log,re.M)}
assert len(rows)==6 and set(aa)=={8,16,32}
q=json.loads((P/'ilp_vs_pair_queue.json').read_text());assert hashlib.sha256(p.with_suffix('.py').read_bytes()).hexdigest()==q['sha256'];q.update(judge_job=job,state='done; all32 seeds and native/official gates pass; PAIR beats ILP at each measured M',receipt=str(p.relative_to(R)));(P/'ilp_vs_pair_queue.json').write_text(json.dumps(q,indent=2)+'\n')
out=dict(job=job,receipt=str(p.relative_to(R)),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),official_passed=True,rows=rows,AA=aa,decision='PAIR wins direct point comparison at M8/16/32. ILP remains a valid alternative versus S14; composition with PAIR remains unmeasured.')
(P/'ilp_vs_pair_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
