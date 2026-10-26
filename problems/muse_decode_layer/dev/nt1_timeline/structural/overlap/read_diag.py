"""Read the bounded clock receipt;keep observer tax anddiagnostic limitations with every table."""
import argparse,json,re,sys
from pathlib import Path
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D.parents[5]/'tools'));from lab_packet import result_lines
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();host=json.loads(a.receipt.read_text());log='\n'.join(result_lines(host['result']['log_tail']));assert host['result']['passed'];blobs=re.findall(r'^P3A_OVERLAP_DIAG (\S+)$',log,re.M);assert len(blobs)==1
n={};exec(compile((D/'diagnostic.py.inc').read_text(),'<overlapdiag>','exec'),n);r=n['overlap_diag_unpack'](blobs[0]);r['job']=host['id'];r['correctness_line']=next(l for l in log.splitlines() if l.startswith('GATE '));r['source_audit']=json.loads((D/'results.json').read_text())['source_audit'];assert r['source_audit']['verified']
for v in r['records']:
 print(v['M'],v['kind'],'observer',v['observer_ratio'],'SE',v['observer_SE'],'MHz',v['clock_MHz'])
 for w,row in enumerate(v['warp_rows']):print(w,[round(t/1000,3) for t in row[:7]],'tiles/segments/copies',row[7:])
 overlaps=[]
 for k,issue in enumerate(v['first8_issue_ns']):
  overlaps.append([j for j,(b,e) in enumerate(zip(v['first8_math_begin_ns'],v['first8_math_end_ns'])) if issue is not None and b is not None and e is not None and b<=issue<=e])
 v['issue_events_inside_consumer_warp0_math_intervals']=overlaps
 print('issue inside math intervals',overlaps)
(D/'clock_results.json').write_text(json.dumps(r,indent=2)+'\n')
