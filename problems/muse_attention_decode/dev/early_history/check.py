"""Fresh data, strided producer partials and repeated captured counter reset."""
import sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE));import patch
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode'),str(HERE.parent/'cluster_sliding')]
import torch,driver
ns={'torch':torch};exec(patch.cell(),ns)
mode=sys.argv[1] if len(sys.argv)>1 else 'early'
if mode=='late':ns={'torch':torch};exec(patch.cell(False),ns)
original=ns['attention']
def call(data,arg=None,**kw):return original(data,arg,nk=128,cs=16,qt=int(sys.argv[2]) if len(sys.argv)>2 else 4)
driver.attention=call
s=(HERE.parent/'cluster_sliding/check.py').read_text()
s=s.replace("if '--quick' in sys.argv:CASES=CASES[:1]+CASES[2:3]+CASES[-6:-5]+CASES[-3:-2]", "CASES=[dict(batch=1,ctx=c,heads='tp8',layer='sliding',page=p,dist=d,kv_scale=sc) for c,p,d,sc in ((10,64,'soft',1),(2047,64,'soft',1),(6000,1,'edges',1),(80000,1,'soft',1),(4000,1,'soft',.015625))]")
s=s.replace('for nk in (64,128):','for nk in (128,):')
s=s.replace('copy_into(data,fresh);graph.replay();torch.cuda.synchronize()', "copy_into(data,fresh)\n            for _ in range(8):graph.replay()\n            torch.cuda.synchronize()\n            assert all(int(torch.count_nonzero(c))==0 for c in ns['_COUNTERS'].values())")
exec(compile(s,str(HERE.parent/'cluster_sliding/check.py'),'exec'),{'__name__':'__main__','__file__':str(HERE.parent/'cluster_sliding/check.py'),'ns':ns})
