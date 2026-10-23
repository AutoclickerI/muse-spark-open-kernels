import sys, importlib.util, json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
sys.path.insert(0, 'problems/muse_prefill_layer')
import cand_attention_fp8 as m

def data_for(B,L,S,page=1,window=0,seed=9917,ks=1.,soft=False):
    g=torch.Generator(device='cuda').manual_seed(seed)
    maxp=triton.cdiv(S+L,page)
    table=torch.randperm(B*maxp,generator=g,device='cuda',dtype=torch.int64).to(torch.int32).view(B,maxp)
    k=torch.randn((B*maxp,page,2,64),generator=g,device='cuda')
    k=k*torch.rsqrt(k.square().mean(-1,keepdim=True))
    if soft:
        u=torch.randn((2,64),generator=g,device='cuda')
        u=u*torch.rsqrt(u.square().mean(-1,keepdim=True))
        k=(.56**.5)*u+(.44**.5)*k
    v=torch.randn(k.shape,generator=g,device='cuda')
    q=torch.randn((B*L,16,64),generator=g,device='cuda')
    q=(q*torch.rsqrt(q.square().mean(-1,keepdim=True))).to(torch.float8_e4m3fn).reshape(B*L,1024)
    data=dict(config=dict(tokens=B*L,batch=B,q_len=L,window=window,page_size=page,
              sm_scale=1/8,qk_scale_factor=5.4730064863875,kv_scale=ks),
              page_table=table,seq_lens=torch.tensor([S-i*min(71,S) for i in range(B)],device='cuda',dtype=torch.int32),
              k_cache=(k/ks).to(torch.float8_e4m3fn),v_cache=(v/ks).to(torch.float8_e4m3fn),
              k_scale=torch.tensor(ks,device='cuda'),v_scale=torch.tensor(ks,device='cuda'))
    return data,q

def ref_rows(data,q):
    cfg=data['config']; B,L=cfg['batch'],cfg['q_len']; page=cfg['page_size']
    # Sample enough rows, including both ends of the causal block; every sampled row attends all its keys.
    idx=torch.linspace(0,L-1,min(17,L),device='cuda').long().unique()
    refs=[]; got=[]
    for b in range(B):
      seq=int(data['seq_lens'][b]); j=torch.arange(seq+L,device='cuda')
      phys=data['page_table'][b,j//page].long()*page+j%page
      k=data['k_cache'].view(-1,2,64)[phys].float()*data['k_scale']
      v=data['v_cache'].view(-1,2,64)[phys].float()*data['v_scale']
      qq=q.view(B,L,16,64)[b,idx].float().transpose(0,1)
      k=k.repeat_interleave(8,1).transpose(0,1)
      v=v.repeat_interleave(8,1).transpose(0,1)
      mask=j[None,:] <= (seq+idx)[:,None]
      if cfg['window']: mask &= j[None,:] > (seq+idx-cfg['window'])[:,None]
      ref=F.scaled_dot_product_attention(qq,k,v,attn_mask=mask[None],scale=cfg['sm_scale']*cfg['qk_scale_factor'])
      refs.append(ref.transpose(0,1))
      got.append((b,idx))
    return got,refs

if __name__ == '__main__':
    variants=[dict(bq=16,bn=128,group=8,warps=8,stages=2),
              dict(bq=16,bn=256,group=8,warps=8,stages=2),
              dict(bq=32,bn=128,group=4,warps=8,stages=2),
              dict(bq=64,bn=128,group=1,warps=4,stages=2)]
    results=[]
    for B,L,S,page,W in [(1,753,80000,1,0),(1,16384,32768,1,0),(1,16384,65536,1,0),
                         (1,257,1025,64,0),(2,139,2053,1,2048)]:
        data,q=data_for(B,L,S,page,W,ks=(1/64 if page==64 else 1.))
        at,refs=ref_rows(data,q)
        for kw in variants:
          try:
            f=lambda:m._attention_fp8(data,q,{},**kw)
            out=f(); torch.cuda.synchronize()
            errs=[]
            for (b,idx),ref in zip(at,refs):
              got=out.view(B,L,16,64)[b,idx].float()
              err=(got-ref).norm()/ref.norm()
              errs.append(float(err)); assert err < .04,err
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
              graph=torch.cuda.CUDAGraph()
              with torch.cuda.graph(graph): replay_out=f()
            torch.cuda.current_stream().wait_stream(stream)
            times=[]
            for _ in range(3):
              a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
              a.record(); graph.replay(); z.record(); z.synchronize(); times.append(a.elapsed_time(z)*1000)
            row=dict(shape=[B,L,S,page,W],variant=kw,relative_l2=max(errs),local_sm120_us=sorted(times)[1])
            print(row,flush=True); results.append(row)
          except Exception as e:
            print('FAILED',B,L,S,kw,type(e).__name__,str(e),flush=True)
        del data,q,out,refs,graph,replay_out
        torch.cuda.empty_cache()
    Path('problems/muse_prefill_layer/dev/attention_fp8_probe.json').write_text(json.dumps(results,indent=2))
