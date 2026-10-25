"""Rank component aggregate gaps once this batch's exact-SHA receipts arrive.
Leader per-shape times are not published in board snapshots; never fabricate them.
"""
import argparse,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
a=argparse.ArgumentParser();a.add_argument('--boards',type=Path,default=R/'runs/boards/20261025T150149Z');args=a.parse_args()
rows=[]
for c in json.loads((H/'manifest.json').read_text())['candidates']:
 receipts=[]
 for p in (R/'runs'/c['problem']).glob('*.json'):
  d=json.loads(p.read_text())
  if d.get('content_hash')==c['sha256'] or d.get('sha256')==c['sha256']:receipts.append((p,d))
 if not receipts:print(c['role'],'pending exact-SHA receipt');continue
 p,d=receipts[-1];r=d.get('result') or {}
 if not r.get('passed'):print(c['role'],'not passed',r.get('error'));continue
 board=json.loads((args.boards/(c['problem']+'.json')).read_text());leader=board['rows'][0]
 overall=next((x for x in board['rows'] if x['team']=='colinlikescode'),None)
 rows.append(dict(role=c['role'],receipt=str(p.relative_to(R)),our_speedup=r['speedup'],our_score_us=r['score_us'],
    component_leader=leader['team'],leader_speedup=leader['speedup'],time_ratio_to_component_leader=leader['speedup']/r['speedup'],
    time_ratio_to_overall_leader=overall['speedup']/r['speedup'] if overall else None,
    our_shapes=[x for x in r.get('records',[]) if x.get('event')=='shape'],leader_per_shape_available=False))
rows.sort(key=lambda x:x['time_ratio_to_component_leader'],reverse=True)
for row in rows:print(row['role'],f"aggregate gap {row['time_ratio_to_component_leader']:.4f}x",row['receipt'])
if rows:(H/'leader_gap.json').write_text(json.dumps(dict(board_snapshot=str(args.boards),ranking=rows),indent=2)+'\n')
