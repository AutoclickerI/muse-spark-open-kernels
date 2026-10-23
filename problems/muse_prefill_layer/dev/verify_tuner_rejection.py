"""Failure injection at construction, oracle check, graph timing and finalist construction."""
import sys,contextlib,io,json,math
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
p=Path('problems/muse_prefill_layer');m=le.load_module('reject_verify',p/'cand_dense_fusion_portfolio.py')
built={};real=m._ScoreGraph;seen=[]
class Trial:
 def __init__(self,fn):
  self.output=fn();self.choice=self.output.choice
  built[self.choice]=built.get(self.choice,0)+1
  if self.choice=='bad_build':raise RuntimeError('injected graph construction failure')
  if self.choice=='bad_finalist' and built[self.choice]>1:raise RuntimeError('injected finalist construction failure')
 def replay(self):
  if self.choice=='bad_time':raise RuntimeError('injected graph replay failure')
  self.output.add_(0)
def run(c):
 x=torch.ones(1,device='cuda');x.choice=c;return x
def check(out):
 if out.choice=='bad_check':raise RuntimeError('injected independent oracle failure')
 return {'finite':bool(torch.isfinite(out).all())}
m._ScoreGraph=Trial;flush=torch.zeros(1<<30,device='cuda',dtype=torch.uint8)
m._time_left=lambda st:100
log=io.StringIO()
with contextlib.redirect_stdout(log):
 chosen,record=m._score_select(('good','bad_build','bad_check','bad_time','bad_finalist'),'good',run,check,lambda r:None,flush,{},'failure injection')
 assert chosen=='good',(chosen,record)
 g=Trial(lambda:run('good'));bad=Trial(lambda:run('bad_time'))
 times=m._score_times([bad,g],lambda r:None,flush,4)
 assert not math.isfinite(times[0]) and math.isfinite(times[1]),times
text=log.getvalue();assert all(n in text for n in ('construction failure','oracle failure','replay failure','finalist construction failure')),text
print(text,flush=True)
(p/'dev/tuner_rejection_validation.json').write_text(json.dumps(dict(ok=True,selected=chosen,constructed=built,logs=text,phases=['graph construction','independent oracle','timing','finalist construction']),indent=2))
print('All tuner failure phases rejected and continued PASS',flush=True)
