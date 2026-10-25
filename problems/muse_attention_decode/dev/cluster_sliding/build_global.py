"""One launch for decode full/sliding, with nonspinning completion and shared softmax merge weights."""
from pathlib import Path
import hashlib,build_last
HERE=Path(__file__).resolve().parent
MERGE=r'''
  // The last CTA computes one weight per split/row, shared across all 64 value dimensions.
  // Reuse the drained gather ring; no extra allocation or launch is needed.
  float* weights=(float*)K;
  for(int r=w;r<NR;r+=4) {
    float mm=-INFINITY;float2 ml[4];
#pragma unroll
    for(int j=0;j<4;j++) {
      const int c=32*j+lane;
      ml[j]=c<p.ns?gpair(p.part+((int64_t)(group*p.ns+c)*NR+r)*68+64):make_float2(-INFINITY,0.f);
      mm=fmaxf(mm,ml[j].x);
    }
#pragma unroll
    for(int d=16;d;d>>=1)mm=fmaxf(mm,__shfl_xor_sync(0xffffffffu,mm,d));
    const float safe=mm==-INFINITY?0.f:mm;
    float den=0.f,z[4];
#pragma unroll
    for(int j=0;j<4;j++) {z[j]=ex2(ml[j].x-safe);den=fmaf(z[j],ml[j].y,den);}
    den=wsum(den);const float scale=*p.vs/den;
#pragma unroll
    for(int j=0;j<4;j++)if(32*j+lane<p.ns)weights[r*128+32*j+lane]=z[j]*scale;
  }
  __syncthreads();
  for(int r=w;r<NR;r+=4) {
    float x=0.f,y=0.f;
    for(int c0=0;c0<p.ns;c0+=8) {
      float2 ov[8];float z[8];
#pragma unroll
      for(int i=0;i<8;i++) {
        const int c=min(c0+i,p.ns-1);
        ov[i]=gpair(p.part+((int64_t)(group*p.ns+c)*NR+r)*68+2*lane);
        z[i]=weights[r*128+c];
      }
#pragma unroll
      for(int i=0;i<8;i++)if(c0+i<p.ns) {x=fmaf(z[i],ov[i].x,x);y=fmaf(z[i],ov[i].y,y);}
    }
    const float norm=rsqrtf(wsum(x*x+y*y)*(1.f/64.f)+1e-5f);
    const int head=kh*8+sub*H+r;float2 gate=make_float2(0.f,0.f);
    if(p.gate)gate=*(float2*)(KN+1024+(r*64+2*lane)*4);
    *(__nv_bfloat162*)(p.out+((int64_t)b*p.HQ+head)*64+2*lane)=p.gate?
      __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate.x)),y*norm/(1.f+__expf(-gate.y))):__floats2bfloat162_rn(x*norm,y*norm);
  }
'''

def source(shared_weights=True):
    s=build_last.source()
    s=s.replace('csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane])','csbuf[lane]=p.window?rope_cs((float)p.positions[b]*p.inv_freq[lane]):make_float2(1.f,0.f)')
    s=s.replace('lo=max(0,seq-p.window+1)','lo=p.window?max(0,seq-p.window+1):0')
    s=s.replace('key>=seq+tok-window+1','(!window || key>=seq+tok-window+1)')
    s=s.replace('ns>32','ns>128').replace('1..32 splits','1..128 splits')
    if shared_weights:
        a=s.index('  for(int r=w;r<NR;r+=4) {',s.index('// No CTA spins'));b=s.index('  __syncthreads();if(tid==0)atomicExch',a)
        s=s[:a]+MERGE+s[b:]
    return s

if __name__=='__main__':
    s=source();p=HERE/'global_attention.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
