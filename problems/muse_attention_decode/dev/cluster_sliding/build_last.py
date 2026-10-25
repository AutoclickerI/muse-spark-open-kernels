"""Nonspinning last-CTA reduction; every split publishes and retires, one CTA finishes the output."""
import build_gated,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
REDUCE=r'''
  // Publish only live query rows. Each split's partial slot is private.
  const int group=(b*p.HK+kh)*QT+sub,g=lane>>2,t=lane&3;
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=w*16+g+8*rh;
    float den=ls[rh];den+=__shfl_xor_sync(0xffffffffu,den,1);den+=__shfl_xor_sync(0xffffffffu,den,2);
    if(row<NR) {
      float* pp=p.part+((int64_t)(group*p.ns+split)*NR+row)*68;
#pragma unroll
      for(int n=0;n<8;n++)*(float2*)(pp+8*n+2*t)=make_float2(O[n][2*rh],O[n][2*rh+1]);
      if(t==0)*(float2*)(pp+64)=make_float2(m[rh],den);
    }
  }
  __threadfence();__syncthreads();
  __shared__ int last;
  if(tid==0) {
    uint32_t ticket;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(ticket):"l"(p.count+group*32):"memory");
    last=ticket==(uint32_t)(p.ns-1);
  }
  __syncthreads();if(!last)return;
  // No CTA spins or stays resident waiting for another CTA. The release/acquire
  // atomic chain publishes every split, and cg loads bypass stale private L1.
  for(int r=w;r<NR;r+=4) {
    float mx=-INFINITY,total=0.f,x=0.f,y=0.f;
    for(int c0=0;c0<p.ns;c0+=8) {
      float2 ml[8],ov[8];
#pragma unroll
      for(int i=0;i<8;i++) {
        const int c=min(c0+i,p.ns-1);const float* pp=p.part+((int64_t)(group*p.ns+c)*NR+r)*68;
        ml[i]=gpair(pp+64);ov[i]=gpair(pp+2*lane);
      }
      float cm=mx;
#pragma unroll
      for(int i=0;i<8;i++)if(c0+i<p.ns)cm=fmaxf(cm,ml[i].x);
      const float sa=mx==-INFINITY?0.f:ex2(mx-cm);total*=sa;x*=sa;y*=sa;
#pragma unroll
      for(int i=0;i<8;i++)if(c0+i<p.ns) {
        const float z=ml[i].x==-INFINITY?0.f:ex2(ml[i].x-cm);
        total=fmaf(z,ml[i].y,total);x=fmaf(z,ov[i].x,x);y=fmaf(z,ov[i].y,y);
      }
      mx=cm;
    }
    const float scl=*p.vs/total;x*=scl;y*=scl;
    const float norm=rsqrtf(wsum(x*x+y*y)*(1.f/64.f)+1e-5f);
    const int head=kh*8+sub*H+r;float2 gate=make_float2(0.f,0.f);
    if(p.gate)gate=*(float2*)(KN+1024+(r*64+2*lane)*4);
    *(__nv_bfloat162*)(p.out+((int64_t)b*p.HQ+head)*64+2*lane)=p.gate?
      __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y))):__floats2bfloat162_rn(x*norm,y*norm);
  }
  __syncthreads();if(tid==0)atomicExch(p.count+group*32,0);
'''

def source():
    s=build_gated.source()
    s=s.replace('int qt, ns, cs, qwidth;', 'int* count; int qt, ns, cs, qwidth;')
    a=s.index('  // All asynchronous use has drained;');b=s.index('\n}\n\ntemplate<int L,int QT,int NK> void attrs()',a)
    s=s[:a]+REDUCE+s[b:]
    a=s.index('DI void rescale(float (&O)')
    s=s[:a]+'''DI float2 gpair(const float* p) {float2 v;asm volatile("ld.global.cg.v2.f32 {%0,%1},[%2];":"=f"(v.x),"=f"(v.y):"l"(p):"memory");return v;}
'''+s[a:]
    a=s.index('  cudaLaunchAttribute at[2]={};');b=s.index('  auto e=cudaLaunchKernelEx',a)
    s=s[:a]+'''  cudaLaunchAttribute at={};at.id=cudaLaunchAttributeProgrammaticStreamSerialization;at.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&at;cfg.numAttrs=1;
'''+s[b:]
    s=s.replace('int64_t width,double sc,int64_t stream) {','int64_t width,double sc,int64_t stream,int64_t part,int64_t counter,int64_t ns) {')
    s=s.replace('  VP p={};p.qkv=', '  if(L!=1 || ns<1 || ns>32)throw std::runtime_error("last CTA sliding requires decode and 1..32 splits");\n  VP p={};p.part=(float*)part;p.count=(int*)counter;p.qkv=')
    s=s.replace('p.qt=qt;p.ns=p.cs=cs;', 'p.qt=qt;p.ns=ns;p.cs=1;')
    # Do not instantiate unused verify kernels.
    s=s.replace('  else {if(qt==4){RES(8,4);}else{RES(8,1);}}', '  else throw std::runtime_error("decode only");')
    s=s.replace('  else {if(qt==4){GO(8,4);}else{GO(8,1);}}','')
    assert 'cload(part' not in s and 'cbar();' not in s
    return s

if __name__=='__main__':
    s=source();p=HERE/'last_sliding.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
