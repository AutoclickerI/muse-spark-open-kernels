"""Official board task checks on every fused-pool/sparse arm, GPU0 image only."""
from pathlib import Path
import sys,json
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools')]
import local_eval
arms={'w1k8':(1,8,1,1,False),'w1k8sp':(1,8,1,1,True),'w2k8':(2,8,1,1,False),'w2k8sp':(2,8,1,1,True),
      'w1k4raw':(1,4,1,1,False),'w1k4sp':(1,4,1,1,True),'w1k4s4':(1,4,4,2,False),'w1k4s4sp':(1,4,4,2,True)}
name=sys.argv[1];vector=name.endswith('vec');name=name.removesuffix('vec');loader=local_eval.load_module

def load(label,path):
 m=loader(label,path)
 if label=='submission':
  m.SP_ARM=arms[name];m.SP_FORCE=True
  if vector:
   original=m.sp_prepare
   m.sp_prepare=lambda *a,**kw:original(*a,**kw,vector=True)
 return m
local_eval.load_module=load
# Kernel source is generated through reviewed template adapters. compile_native.py
# covers every final instantiated native unit; local_eval still checks Python.
out=H/'screen'/(name+('vec' if vector else ''));out.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'candidate.py'),'--no-baseline','--reps','2','--inputs','3','--json',str(out/'result.json')]
local_eval.main()
