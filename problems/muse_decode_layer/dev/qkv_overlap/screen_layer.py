from pathlib import Path
import ast,sys,json,time,hashlib
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(HERE.parent),str(HERE)]
import local_eval,verify_best
from runtime import KVGroupOverlap

def main():
    torch.backends.cuda.matmul.allow_tf32=False;started=time.monotonic()
    lab=local_eval.load_module('coop_helpers',ROOT/'prof/lab/best7_coop_glue.py')
    m=lab.dp_module('group_overlap_whole_layer',lab.BEST7_SOURCE);calls=verify_best.install(m,False)
    text=(HERE/'best7_kv_group_cell.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_kg_best7_run')
    fn='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n';exec(compile(fn,'<group_overlap_layer>','exec'),m._ns)
    experts=lab.cg_experts('cuda');weights=lab.fg_weights(m,experts)
    state=lab.rg_setup(m,weights,dict(native=lab.cg_bank(m,experts)),time.monotonic()+250)
    state['kg_ops']={};rows=[]
    for b in (1,16,32):
      for window in (2048,0):
        data=m._ap_generate_input(batch=b,ctx=80000,layer='sliding' if window else 'full',seed=172903,q_len=1,
            spread='agentic' if b>1 else 'fixed',dist='soft',page=1,device='cuda')
        plan=lab.rg_plan(m,b,1,window);op=KVGroupOverlap(state['qkv_gate'],b,2048 if window else 80000)
        op.configure(data['config']['rope_theta']);state['kg_ops'][b,window]=op
        a,ao=lab.dp_graph(lambda:m._best7_run(data,state,plan))
        bg,bo=lab.dp_graph(lambda:m._kg_best7_run(data,state,plan))
        original=data['stream'].clone()
        for seed in lab.DP_SEEDS:
            g=torch.Generator(device='cuda').manual_seed(seed)
            data['stream'].copy_(original+torch.randn(original.shape,device='cuda',generator=g)*.2)
            data['attn_input'].copy_(m._ap_rms(data['stream']).bfloat16());data['local_start'].fill_(32*(seed%8))
            a.replay();bg.replay();n=int((ao[2]!=bo[2]).any(-1).sum())
            row=dict(batch=b,window=window,seed=seed,flips=n)
            if n:row.update(disabled=True,detail=lab.fg_flip_report(m,data,state,plan,bo[2],ao[2]))
            else:
                try:row['l2']=lab.fg_layer_gate(ao,bo,data,state)[0]
                except RuntimeError as e:raise RuntimeError(f'case b={b} window={window} seed={seed}: {e}') from e
            rows.append(row)
    result=dict(cell_sha256=hashlib.sha256(text.encode()).hexdigest(),checks=len(rows),rows=rows,
        flips=sum(r['flips'] for r in rows),max_l2=max((r.get('l2',0) for r in rows)),elapsed_s=time.monotonic()-started)
    (HERE/'layer_validation.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='rows'})
if __name__=='__main__':main()
