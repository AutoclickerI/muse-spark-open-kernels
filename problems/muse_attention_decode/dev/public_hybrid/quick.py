import argparse,gc,importlib.util,json,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;P=H.parent/'public_ports';R=H.parents[3];sys.path[:0]=[str(R/'problems/muse_attention_decode')]
import reference as ref
ref._fa3_scheduler_metadata=lambda *a,**kw:None
ap=argparse.ArgumentParser();ap.add_argument('--lee',action='store_true');a=ap.parse_args()
def mod(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
m=mod('hybrid_quick',H/'out'/('lee_hybrid.py' if a.lee else 'operator_direct.py'))
s=mod('public_stands',P/'local_standins.py')
if a.lee:s.lee(m)
else:s.operator(m,False)
m.setup('cuda');rows=[]
for l in (1,8):
 for b in (1,16,32):
  for w in ((2048,0) if a.lee else (2048,)):
   d=ref.generate_input(batch=b,q_len=l,ctx=257,page=1,heads='tp8',layer='sliding' if w else 'full',dist='edges',seed=9815723+b+l+w)
   for sp,width in ((1,1280),(3,2304)):
    p=torch.zeros((sp,b*l,width),device='cuda');p[0].copy_(d['qkv'][...,:width].reshape(b*l,width).float());rd=dict(d);rd['qkv']=d['qkv'].clone()
    if width==1280:rd['qkv'][...,1280:]=float('inf')
    for arm in (('F','K') if l==1 and not w and b>1 and a.lee else ('F','P','Q','E') if l==8 and not w and a.lee else ('F',)):
     y=m.attention_pair((p,sp),rd,arm) if a.lee else m.attention_pair((p,sp),rd);torch.cuda.synchronize();ok,msg=ref.check_implementation(rd,ref.collect(rd,y));print(b,l,w,sp,width,arm,ok,msg,flush=True);assert ok,msg;rows.append([b,l,w,sp,width,arm,ok])
   d=p=y=None;gc.collect();torch.cuda.empty_cache()
(H/'out'/('quick_lee.json' if a.lee else 'quick_operator.json')).write_text(json.dumps(rows)+'\n')
