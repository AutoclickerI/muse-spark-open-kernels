import triton
import triton.language as tl
@triton.jit
def pack5_encode(P,Q,O,LO,RAW,MASK,N:tl.constexpr,B:tl.constexpr):
 tile=tl.program_id(0);packet=tl.program_id(1);i=tl.arange(0,B)
 offset=tl.load(O+tile);base=tl.load(LO+tile*16+packet);raw=tl.load(RAW+tile*16+packet);mask=tl.load(MASK+tile)
 # Each of16 programs copies one disjoint2048-byte weight slice.
 v=tl.load(P+tile*36864+packet*2048+i,i<2048,0)
 tl.store(Q+offset+packet*2048+i,v,i<2048)
 if packet==0:tl.store((Q+offset+32784).to(tl.pointer_type(tl.uint32)),mask)
 tl.store(Q+offset+32768+packet,base)
 prev=tl.extra.cuda.libdevice.popc(mask&((1<<packet)-1))
 dest=offset+32800+packet*160+prev*96
 src=tile*36864+32768+packet*256
 if raw:
  x=tl.load(P+src+i,i<256,0);tl.store(Q+dest+i,x,i<256)
 else:
  bit=i*8;j=bit//5;shift=bit%5
  x0=tl.load(P+src+j,(i<160)&(j<256),0).to(tl.uint32)-base
  x1=tl.load(P+src+j+1,(i<160)&(j+1<256),0).to(tl.uint32)-base
  x2=tl.load(P+src+j+2,(i<160)&(j+2<256),0).to(tl.uint32)-base
  out=(x0>>shift)|(x1<<(5-shift))|(x2<<(10-shift))
  tl.store(Q+dest+i,out.to(tl.uint8),i<160)
def pack5_bank(t):
 assert t.dtype==torch.uint8
 old=t.reshape(-1,36864);n=old.shape[0]
 values=old[:,32768:].reshape(n,16,256).to(torch.int16)
 lo=values.amin(2).to(torch.uint8);hi=values.amax(2)
 raw=(hi-lo.to(torch.int16)>31).to(torch.int32)
 mask=(raw*(1<<torch.arange(16,device=t.device,dtype=torch.int32))).sum(1).to(torch.int32)
 sizes=(32800+(raw*96+160).sum(1)).to(torch.int64)
 header=((n+1)*8+15)//16*16
 offsets=torch.cat((torch.zeros(1,device=t.device,dtype=torch.int64),sizes.cumsum(0)))+header
 total=int(offsets[-1]);out=torch.zeros(total,device=t.device,dtype=torch.uint8)
 out[:(n+1)*8].view(torch.int64).copy_(offsets)
 pack5_encode[(n,16)](old,out,offsets,lo,raw,mask,n,2048,num_warps=8)
 return out
