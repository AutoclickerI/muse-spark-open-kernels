"""Pin the new integration base to the N16G4 control used by the cluster lab."""
import ast
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
from light_build import value


def fun(s,name):
 return ast.get_source_segment(s,next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name))


def main():
 p=R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py';s=p.read_text()
 assert hashlib.sha256(s.encode()).hexdigest()=='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
 lab=(H/'cluster_payload.py').read_text()
 lc=value(lab,'OP_CUDA')[1];assert value(s,'NF_CUDA')==lc
 assert value(s,'NF_CPP')==value(lab,'OP_CPP').split('void oc_gather(')[0]
 lp=fun(lab,'nf_light_plan').replace('def nf_light_plan(data,base,native,streams,light_grid):','def nf_plan(data,base,native,streams):\n    light_grid=4')
 assert ast.dump(ast.parse(lp))==ast.dump(ast.parse(fun(s,'nf_plan')))
 same=[]
 for name in ('nf_perm','nf_ilv','nf_wide'):
  assert ast.dump(ast.parse(fun(s,name)))==ast.dump(ast.parse(fun(lab,name)));same.append(name)
 packets={}
 for name in ('cluster_copy_payload.py','cluster_schedule_payload.py'):
  t=(H/name).read_text()
  assert value(t,'OP_CUDA')[0]==lc
  assert value(t,'OP_CPP').split('void oc_gather(')[0]==value(s,'NF_CPP')
  for f in ('nf_light_plan','nf_perm','nf_ilv','nf_wide'):
   assert ast.dump(ast.parse(fun(t,f)))==ast.dump(ast.parse(fun(lab,f)))
  packets[name]=hashlib.sha256(t.encode()).hexdigest()
 out=dict(base=str(p.relative_to(R)),base_sha256=hashlib.sha256(s.encode()).hexdigest(),
          nf_cuda_sha256=hashlib.sha256(lc.encode()).hexdigest(),nf_cpp_equal=True,nf_plan_AST_equal=True,
          bank_transforms_AST_equal=same,successor_controls=packets,note='N16G4 control is exactly the S14 nf CUDA/plan and bank transformations. New integration patches must pin S14, not S12.')
 (H/'s14_cell_equiv.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

if __name__=='__main__':main()
