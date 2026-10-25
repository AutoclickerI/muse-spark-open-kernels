from pathlib import Path
import importlib.util,torch,time,json,hashlib
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('verify_dense_projection',HERE/'projection.py');vd=importlib.util.module_from_spec(spec);spec.loader.exec_module(vd)

def portable(self,x):
    self._call(x,False)  # Exercise exact native shape/layout/workspace conversion.
    torch.mm(x,self.w.T,out=self.out) if self.out.dtype==torch.bfloat16 else torch.mm(x,self.w.T,out=self.out,out_dtype=torch.float32)
    return self.out


def weights(device='cuda'):
 g=torch.Generator(device=device).manual_seed(941732)
 result={name:(torch.randn((n,k),device=device,generator=g)/k**.5).bfloat16() for name,(n,k) in dict(qkv_gate=(2304,8192),o=(8192,1024),unpooler=(8192,4096)).items()}
 pool=(torch.randn((4096,8192),device=device,generator=g)/8192**.5).bfloat16()
 router=torch.randn((256,8192),device=device,generator=g)/8192**.5;hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
 result['pool_router']=torch.cat((pool,hi,lo));return result,router


def main():
 started=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
 vd.VDCutlass.__call__=portable
 ws=torch.empty(1<<30,device='cuda',dtype=torch.uint8);banks,router=weights();gen=torch.Generator(device='cuda').manual_seed(712341)
 rows=[]
 for m in (64,128,256):
  for role,w in banks.items():
   fp32=role=='pool_router';x=torch.randn((m,w.shape[1]),device='cuda',dtype=torch.bfloat16,generator=gen)
   lt=vd.VDLt(w,m,fp32,extended=False,workspace=ws)
   nn=vd.VDLt(w,m,fp32,nn=1,extended=False,workspace=ws)
   candidates=[('lt_tn',lt),('lt_nn',nn),('deep_gemm',vd.VDDeepGemm(w,m,fp32))]
   for code in range(3):candidates.append((f'cutlass{code}',vd.VDCutlass(w,m,fp32,code)))
   for label,op in candidates:
    for _ in range(2):op(x)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=op(x)
    for draw in range(3):
     x.normal_(generator=gen)
     if draw==1:x[:,:w.shape[1]//100]*=16
     if draw==2:x.zero_();x[:,17]=16
     graph.replay();ref=x.float()@w.float().T;d=out.float()-ref
     error=float(d.norm()/ref.norm().clamp_min(1e-12));row=float((d.norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).max());lim=1e-4 if fp32 else .01
     assert torch.isfinite(out).all() and error<lim and row<lim*5,(m,role,label,draw,error,row)
     flips=0;router_error=0.
     if fp32:
      score=(out[:,4096:4352]+out[:,4352:]).float();target=x.float()@router.T
      router_error=float((score-target).norm()/target.norm().clamp_min(1e-12));assert router_error<1e-4,(m,label,router_error)
      bias=torch.arange(256,device='cuda').float()*1e-4
      got=(score*.1767766952966369).sigmoid()+bias;want=(target*.1767766952966369).sigmoid()+bias
      flipped=(got.topk(8,-1).indices!=want.topk(8,-1).indices).any(-1);flips=int(flipped.sum())
      if flips:
       top=want.topk(9,-1).values;delta=(got-want).abs().amax(-1)
       assert ((top[:,7]-top[:,8])[flipped]<=2*delta[flipped]+1e-6).all(),(m,label,'not near tie')
     rows.append(dict(m=m,role=role,cell=label,draw=draw,l2=error,row=row,flips=flips,router_l2=router_error,native_cutlass=False))
 result=dict(projection_sha256=hashlib.sha256((HERE/'projection.py').read_bytes()).hexdigest(),checks=len(rows),rows=rows,route_flips=sum(r['flips'] for r in rows),max_bf16_l2=max(r['l2'] for r in rows),max_router_l2=max(r['router_l2'] for r in rows),cutlass='SM90a native compile + exact argument/workspace preparation; portable device math only on sm120',elapsed_s=time.monotonic()-started)
 (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='rows'})
if __name__=='__main__':main()
