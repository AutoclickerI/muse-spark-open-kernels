"""Force shipped S14 Hopper attention glue; keep native NT1/NF math and all reference/dispatch checks.
The historical best6 force runner also demanded timing receipts for retired optional adapters. This runner
uses its underlying attention stand-ins directly, so validation covers the12 shipped S14 dispatch rows.
"""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];DEV=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev')]
import local_eval,verify_best
candidate=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();original=local_eval.load_module;seen={};metadata={}
def loader(name,path):
 m=original(name,path)
 if name!='submission':return m
 metadata['attention_standins']=verify_best.install(m,False)
 small=m.smallm_experts;nf=m.nf_call
 def counted(d,st,f,stages=2,grid=264):
  mm=d['hidden_states'].shape[0];assert f==m.SMALLM_FLAGS[mm] and (stages,grid)==(2,264)
  key='NT1:'+str(mm);row=seen.setdefault(key,{'eager':0,'captured':0});row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
  return small(d,st,f,stages,grid)
 def counted_nf(d,st):
  mm=d['hidden_states'].shape[0];assert mm in (128,256);key='NF:'+str(mm);row=seen.setdefault(key,{'eager':0,'captured':0});row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
  return nf(d,st)
 m.smallm_experts=counted;m.nf_call=counted_nf
 return m
local_eval.load_module=loader
spec=importlib.util.spec_from_file_location('s14_shipped_audit',DEV/'audit_s14.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
sys.argv=['audit.py',str(candidate),'--out',str(out)]
try:audit.main()
except SystemExit as e:
 if e.code==0:
  assert set(seen)=={'NT1:1','NT1:8','NT1:16','NT1:32','NF:128','NF:256'},seen
  assert all(v['eager'] and v['captured'] for v in seen.values()),seen
  assert all(metadata['attention_standins'].get(k,0)>0 for k in ('attn_main','xqa')),metadata
  out.mkdir(parents=True,exist_ok=True);(out/'forced_shipped_coverage.json').write_text(json.dumps({'native_math_calls':seen,**metadata},indent=2)+'\n')
  print('FORCED_SHIPPED_COVERAGE',json.dumps(seen),flush=True)
 raise
