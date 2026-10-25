"""Memcheck the TMA barrier/activation-ring lifetimes using changed-input captured replays."""
from pathlib import Path
DEV=Path(__file__).resolve().parent
src=(DEV/'check_wgq.py').read_text().split('checks=0;')[0]
exec(compile(src,str(DEV/'check_wgq.py'),'exec'))
for m in (8,16,32):
 d=n['fx_make_fixture'](w,m,9971);graphs={}
 for mode in (4,8,12):
  c['wgq_experts'](d,st,mode);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=c['wgq_experts'](d,st,mode)
  graphs[mode]=(g,out)
 for draw in range(4):
  fresh=n['fx_make_fixture'](w,m,9981+31*draw)
  for k in ('hidden_states','topk_ids','topk_weights'):d[k].copy_(fresh[k])
  want=n['nt1_fp32'](d,w)
  for mode,(g,out) in graphs.items():
   g.replay();torch.cuda.synchronize();ok,err,row=n['task_check'](out,want)
   assert ok,(m,draw,mode,err,row)
 print('MEMCHECK_PASS',m,flush=True)
print('MEMCHECK_END 36 changed-input replays; cell='+hashlib.sha256((DEV/'wgq_cell.py').read_bytes()).hexdigest(),flush=True)
