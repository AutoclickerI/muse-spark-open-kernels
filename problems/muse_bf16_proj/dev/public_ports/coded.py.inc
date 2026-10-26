# Exact 12-bit payload plus one i32 escape offset per128 weights. Escape nibble15
# stores a local0..127 index in its otherwise-unused eight low bits. Any BF16 bit
# pattern is representable, including dense escape groups; no overflow fallback.
def pp_encode(w):
 start=time.monotonic();n,k=w.shape;b=w.contiguous().view(torch.int16).to(torch.int32)&65535;e=(b>>7)&255;r=n//16
 hist=torch.zeros((r,256),device=w.device,dtype=torch.int32);hist.scatter_add_(1,e.view(r,16*k).long(),torch.ones((r,16*k),device=w.device,dtype=torch.int32));hist[:,0]=0;hist[:,255]=0;cs=torch.nn.functional.pad(hist.cumsum(1),(1,0));wins=(cs[:,15:]-cs[:,:-15])[:,1:241];base=wins.argmax(1).to(torch.int32)+1;delta=e-base.repeat_interleave(16)[:,None];escape=(delta<0)|(delta>14);rank=escape.reshape(-1,128).cumsum(1).reshape(n,k)-1
 low=((b>>8)&128)|(b&127);low=torch.where(escape,rank,low).to(torch.int64);d=torch.where(escape,15,delta).to(torch.int64);v=low.view(n,k//8,8);a=sum(v[:,:,i]<<(8*i) for i in range(4));bb=sum(v[:,:,i+4]<<(8*i) for i in range(4));ex=sum(d.view(n,k//8,8)[:,:,i]<<(4*i) for i in range(8));words=torch.stack((a,bb,ex),dim=2).to(torch.int32);counts=escape.view(-1,128).sum(1);offset=torch.nn.functional.pad(counts.cumsum(0),(1,0)).to(torch.int32);values=torch.nn.functional.pad(b[escape].to(torch.int16),(0,1));torch.cuda.synchronize()
 out=dict(words=words,base=base,offset=offset,values=values,n=n,k=k,prepare_s=time.monotonic()-start);assert torch.equal(pp_decode(out).view(torch.int16),w.view(torch.int16)),'C12 bit roundtrip';return out

def pp_decode(p):
 n,k=p['n'],p['k'];c=p['words'].long()&4294967295;idx=torch.arange(k,device=c.device);j=idx%8;low=torch.where(j<4,c[:,:,0].repeat_interleave(8,1),c[:,:,1].repeat_interleave(8,1));low=(low>>((j%4)*8)[None,:])&255;d=(c[:,:,2].repeat_interleave(8,1)>>(j*4)[None,:])&15;bits=((low&128)<<8)|((p['base'].repeat_interleave(16)[:,None].long()+d)<<7)|(low&127);mask=d==15;offs=p['offset'][:-1].view(n,k//128).repeat_interleave(128,1).long();bits[mask]=p['values'][(offs+low)[mask]].long()&65535;return bits.to(torch.int16).view(torch.bfloat16)

def pp_slice(p,lo,hi):
 k=p['k'];return dict(p,words=p['words'][lo:hi],base=p['base'][lo//16:hi//16],offset=p['offset'][lo*(k//128):hi*(k//128)+1],n=hi-lo)

@triton.jit
def _pp_c12(W,BASE,OFF,ESC,r,k0,K:tl.constexpr,BK:tl.constexpr):
 # Read each packed word once before expanding its eight decoded weights.
 # Repeated per-weight loads otherwise become oversized software-pipeline slots.
 kg=k0//8+tl.arange(0,BK//8);group=r[:,None]*(K//8)+kg[None,:]
 lo0=tl.load(W+group*3);lo1=tl.load(W+group*3+1);ex=tl.load(W+group*3+2);lane=tl.arange(0,8)
 low=(tl.where(lane[None,None,:]<4,lo0[:,:,None],lo1[:,:,None])>>((lane[None,None,:]%4)*8))&255
 d=(ex[:,:,None]>>(lane[None,None,:]*4))&15;low=tl.reshape(low,(r.shape[0],BK));d=tl.reshape(d,(r.shape[0],BK));base=tl.load(BASE+r//16)
 bits=((low&128)<<8)|((base[:,None]+d)<<7)|(low&127)
 eg=k0//128+tl.arange(0,BK//128);eo=tl.load(OFF+r[:,None]*(K//128)+eg[None,:]);eo=tl.reshape(tl.broadcast_to(eo[:,:,None],(r.shape[0],BK//128,128)),(r.shape[0],BK))
 ev=tl.load(ESC+eo+low,mask=d==15,other=0).to(tl.int32)&65535;bits=tl.where(d==15,ev,bits);return bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)

@triton.jit
def _pp_coded(XD,W,BASE,OFF,ESC,P,M,N,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,PDL:tl.constexpr=True):
 if PDL:tl.extra.cuda.gdc_launch_dependents();tl.extra.cuda.gdc_wait()
 pm=tl.program_id(0);pn=tl.program_id(1);ps=tl.program_id(2);r=pn*BN+tl.arange(0,BN);ks=tl.arange(0,BK);steps=K//BK;first=ps*steps//S;last=(ps+1)*steps//S;acc=tl.zeros((BN,BM),tl.float32)
 for kk in range(first,last):
  wt=_pp_c12(W,BASE,OFF,ESC,r,kk*BK,K,BK);xt=XD.load([pm*BM,kk*BK]).T;acc=tl.dot(wt,xt,acc)
 mt=pm*BM+tl.arange(0,BM);tl.store(P+ps*M*N+mt[None,:]*N+r[:,None],acc,mask=mt[None,:]<M)
