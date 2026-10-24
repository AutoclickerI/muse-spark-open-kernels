from pathlib import Path
import sys,json,time
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(HERE)]
import local_eval
from runtime import KVGroupOverlap

def error(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))

def main():
    torch.backends.cuda.matmul.allow_tf32=False;started=time.monotonic()
    m=local_eval.load_module('overlap_reference_helpers',ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py')
    m._ap_fa3_scheduler_metadata=lambda *a:None
    g=torch.Generator(device='cuda').manual_seed(32537)
    weight=(torch.randn((2304,8192),device='cuda',generator=g)/8192**.5).bfloat16();rows=[]
    for b in (1,16,32):
      for window in (2048,0):
       for rope in (False,True):
        data=m._ap_generate_input(batch=b,ctx=4096,layer='sliding' if window else 'full',seed=172903,q_len=1,
            spread='agentic' if b>1 else 'fixed',dist='soft',page=1,device='cuda')
        data['config']['rope_theta']=10000. if rope else 0.
        op=KVGroupOverlap(weight,b,4096 if not window else window);op.configure(data['config']['rope_theta'])
        x=data['attn_input'];original=x.clone()
        for _ in range(2):op(x,data,True)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=op(x,data,True)
        graph.replay()
        for draw in range(3):
            x.copy_((original.float()+torch.randn(original.shape,device='cuda',generator=g)*.1).bfloat16())
            # Poison only the output and the new cache rows, to expose stale data.
            op.out.fill_(float('nan'));loc=data['cache_loc'].long()
            data['k_cache'].view(-1,2,64)[loc]=0;data['v_cache'].view(-1,2,64)[loc]=0
            graph.replay();got=out.clone()
            raw=torch.mm(x,weight.T).unsqueeze(0)
            m._attention_store_agrees((raw,1),data)
            ref=m._attention_reference((raw,1),data,False)
            e=error(got,ref);row_error=float(((got.float()-ref.float()).norm(dim=-1)/ref.float().norm(dim=-1).clamp_min(1e-12)).max());assert e<.01 and row_error<.05,(b,window,rope,draw,e,row_error)
            serial=op(x,data,False).clone();se=error(serial,got);assert se==0,(b,window,rope,draw,se)
            rows.append(dict(batch=b,window=window,rope=rope,draw=draw,l2=e,max_row_l2=row_error,serial_l2=se))
    result=dict(checks=len(rows),max_l2=max(r['l2'] for r in rows),graph_changed_inputs=True,rows=rows,elapsed_s=time.monotonic()-started)
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='rows'})
if __name__=='__main__':main()
