@gluon.jit
def _abi_xg_offsets(sorted_ptr, ROWMAP, r0, r1, K:gl.constexpr, BM:gl.constexpr):
 rr=gl.arange(0,4*BM,layout=gl.SliceLayout(1,LXG));cc=gl.arange(0,64,layout=gl.SliceLayout(0,LXG));rv=r0+rr//4
 pair=gl.load(sorted_ptr+rv,mask=rv<r1,other=0);row=gl.load(ROWMAP+pair,mask=rv<r1,other=0)
 return gl.expand_dims(row*K+(rr%4)*64,1)+gl.expand_dims(cc,0),gl.expand_dims(rv<r1,1)&gl.expand_dims(cc>=0,0)

@triton.jit
def _abi_work(C,OFF,W,NW,W13,NW13,M:tl.constexpr,MAXW:tl.constexpr,MAX13:tl.constexpr):
 e=tl.arange(0,32);cnt=tl.load(C+e);off=tl.load(OFF+e)
 c=(cnt+15)//16;pref=tl.cumsum(c)-c;total=tl.sum(c,0);i=tl.arange(0,MAXW)
 pick=(i[:,None]>=pref[None,:])&(i[:,None]<(pref+c)[None,:]);owner=tl.sum(tl.where(pick,e[None,:],0),1);base=tl.sum(tl.where(pick,off[None,:],0),1);first=tl.sum(tl.where(pick,pref[None,:],0),1);size=tl.sum(tl.where(pick,cnt[None,:],0),1);r0=base+(i-first)*16
 tl.store(W+3*i,owner,mask=i<total);tl.store(W+3*i+1,r0,mask=i<total);tl.store(W+3*i+2,tl.minimum(r0+16,base+size),mask=i<total);tl.store(NW,total)
 c=(cnt+31)//32;pref=tl.cumsum(c)-c;total=tl.sum(c,0);i=tl.arange(0,MAX13)
 pick=(i[:,None]>=pref[None,:])&(i[:,None]<(pref+c)[None,:]);owner=tl.sum(tl.where(pick,e[None,:],0),1);base=tl.sum(tl.where(pick,off[None,:],0),1);first=tl.sum(tl.where(pick,pref[None,:],0),1);size=tl.sum(tl.where(pick,cnt[None,:],0),1);r0=base+(i-first)*32
 tl.store(W13+3*i,owner,mask=i<total);tl.store(W13+3*i+1,r0,mask=i<total);tl.store(W13+3*i+2,tl.minimum(r0+32,base+size),mask=i<total);tl.store(NW13,total)

@triton.jit
def _abi_sorted(C,OFF,SLOTS,SORTED,ROWMAP,M:tl.constexpr,B:tl.constexpr):
 e=tl.program_id(0);r=tl.arange(0,B);n=tl.load(C+e);off=tl.load(OFF+e);pair=tl.load(SLOTS+e*M+r,mask=r<n,other=0)
 tl.store(SORTED+off+r,pair,mask=r<n);tl.store(ROWMAP+pair,off+r,mask=r<n)

def abi_plan(data,bank,s19,which='both'):
 # Build only route worklists. S19 owns X, expert metadata, inter, Y, SS and
 # finalize; the GEMMs read/write these buffers with no activation conversion.
 owners=s19['owners'];h13,h2,nv,ids,tw,pos,tasks,l13,l2=owners
 x=h13[3];C,OFF,SLOTS=h13[4],h13[5],h13[8];PNW,ACT,Y,SS=h13[9:13];M=x.shape[0];dev=x.device
 cap=ACT.shape[0];p=M*8;wm=triton.next_power_of_2(triton.cdiv(p,16)+32);w13m=triton.next_power_of_2(triton.cdiv(p,32)+32)
 work=torch.empty((wm,3),dtype=torch.int32,device=dev);work13=torch.empty((w13m,3),dtype=torch.int32,device=dev);nw=torch.empty(1,dtype=torch.int32,device=dev);nw13=torch.empty_like(nw);sorted_=torch.empty(cap,dtype=torch.int32,device=dev);rowmap=torch.empty(p,dtype=torch.int32,device=dev)
 route0=dict(s19['steps'])['route'];base_steps=dict(s19['phase_steps'])
 def route():
  route0();_abi_work[(1,)](C,OFF,work,nw,work13,nw13,M,wm,w13m,num_warps=4);_abi_sorted[(32,)](C,OFF,SLOTS,sorted_,rowmap,M,triton.next_power_of_2(M),num_warps=4)
 def w13():
  _abi_w13_mix[(64,w13m)](x,bank['w13'],bank['w13_sf'],bank['w13_gs'],sorted_,work13,nw13,ACT,K=4096,I=4096,BN=64,BK=256,BM=32,XS=True,SMB=False,HOP=False,CHK=True,PF=False,ROWMAP=rowmap,SS=SS,num_warps=4,maxnreg=128)
 def w2():
  _abi_w2_mix[(64,wm)](ACT,bank['w2'],bank['w2_sf'],bank['w2_gs'],sorted_,work,nw,Y,K=4096,N=4096,BN=64,BK=256,BM=16,XS=True,SMB=False,HOP=False,CHK=True,PF=False,ROWMAP=rowmap,PNW=PNW,SS=SS,num_warps=4)
 phases=[('route',route),('w13',w13 if which in ('both','w13') else base_steps['w13']),('w2',w2 if which in ('both','w2') else base_steps['w2']),('finish',base_steps['finish'])]
 return dict(steps=phases,phase_steps=phases,out=s19['out'],owners=(owners,work,work13,nw,nw13,sorted_,rowmap))
