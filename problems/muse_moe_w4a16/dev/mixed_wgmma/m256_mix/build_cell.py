"""Freeze one composition for decode integration; this does not establish an H200 win."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path.insert(0,str(H.parent))
from build_prefill_lab import subset


def main():
    p=argparse.ArgumentParser();p.add_argument('--w13',nargs=2,type=int,default=(16,32));p.add_argument('--w2',nargs=2,type=int,default=(16,32));p.add_argument('--decoder',type=int,choices=(0,1,2),default=1);args=p.parse_args()
    a=tuple(args.w13);b=tuple(args.w2);assert a[0] in (8,16,32) and a[1] in (32,64);assert b[0] in (0,8,16,32) and b[1] in (32,64)
    source=(H/'adapter.py').read_text();tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('mx_plan','mx_call'):
            node.args.args=node.args.args[:2];node.args.defaults=[]
            if node.name=='mx_plan':
                node.body=ast.parse(f'a={a!r}\nb={b!r}\ndecoder={args.decoder}\ngrid=2112\ncontrol=False\n').body+node.body
            else:
                for n in ast.walk(node):
                    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='mx_plan':n.args=n.args[:2]
    text=subset(ast.unparse(ast.fix_missing_locations(tree)),('mx_prepare','mx_call'))
    tree=ast.parse(text);names={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):names[node.name]='m256_mix_'+node.name
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store):names[n.id]='M256_MIX_'+n.id
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self,n):n.name=names.get(n.name,n.name);return self.generic_visit(n)
        def visit_ClassDef(self,n):n.name=names.get(n.name,n.name);return self.generic_visit(n)
        def visit_Name(self,n):n.id=names.get(n.id,n.id);return n
    text='"""Fixed M256 expert composition. Locally checked; H200 ranking pending."""\n'+ast.unparse(ast.fix_missing_locations(Rename().visit(tree)))+'\n'
    text+='''
def m256_mix_prepare(weights):
    return m256_mix_mx_prepare(weights)

def m256_mix_experts(x,scores,local_ids,state,local_start=0):
    data=dict(hidden_states=x,topk_ids=local_ids,topk_weights=scores,config=dict(local_start=local_start))
    return m256_mix_mx_call(data,state)
'''
    out=H/'unmeasured_cell.py';out.write_text(text)
    meta=dict(path=str(out.relative_to(R)),sha256=hashlib.sha256(text.encode()).hexdigest(),W13=a,W2=b,decoder=args.decoder,grid=2112,
              prepare='m256_mix_prepare(weights)',call='m256_mix_experts(x,scores,local_ids,state,local_start=0)',H200_win=False,status='reviewable fixed port; do not promote as winner without paired H200 evidence',source_payload_sha256=hashlib.sha256((R/'prof/lab/moe_m256_mix.py').read_bytes()).hexdigest())
    (H/'cell.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)


if __name__=='__main__':main()
