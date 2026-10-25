"""Cooperative decode projection reduction: one warp per real Q/K/V row, one input load pass."""
import build,build_warp,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PREP=r'''
template<int QT>
DI void prep_decode(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,float2* csbuf,int b,int kh,int sub,int seq,bool last) {
  constexpr int H=8/QT,R=(H+2+3)/4;
  const int tid=threadIdx.x,lane=tid&31,w=tid>>5;
  // All metadata-only work overlaps the still-running QKV producer.
  for(int i=tid;i<256;i+=128) {
    if(i*16>=H*64)*(uint4*)(Q+i*16)=make_uint4(0,0,0,0);
    *(uint4*)(KN+i*16)=make_uint4(0,0,0,0);*(uint4*)(VN+i*16)=make_uint4(0,0,0,0);
  }
  if(w==0)csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane]);
  __syncthreads();
  asm volatile("griddepcontrol.wait;":::"memory");
  int tk[R],col[R];bool valid[R];float2 x[R];
#pragma unroll
  for(int j=0;j<R;j++) {
    const int row=w+4*j;tk[j]=b;valid[j]=row<H+2;
    const int head=kh*8+sub*H+min(row,H-1);
    col[j]=row<H?head*64+2*lane:row==H?(p.HQ+kh)*64+2*lane:row==H+1?(p.HQ+p.HK+kh)*64+2*lane:0;
  }
  input_pairs<R>(p,tk,col,valid,x);
  const float2 cs=csbuf[lane];
#pragma unroll
  for(int j=0;j<R;j++) {
    const int row=w+4*j;float2 y=x[j];
    if(row<H+1) {
      const float norm=rsqrtf(wsum(y.x*y.x+y.y*y.y)*(1.f/64.f)+1e-5f);
      const float a=y.x*norm,z=y.y*norm;y=make_float2(a*cs.x-z*cs.y,a*cs.y+z*cs.x);
    }
    if(row<H)*(uint16_t*)(Q+off(row,2*lane))=fp8pair(y.x,y.y);
    else if(row<H+2) {
      const uint16_t kv=fp8pair(y.x/(row==H?*p.ks:*p.vs),y.y/(row==H?*p.ks:*p.vs));
      if(row==H)*(uint16_t*)(KN+off(0,2*lane))=kv;
      else {VN[off(2*lane,0)]=(uint8_t)kv;VN[off(2*lane+1,0)]=(uint8_t)(kv>>8);}
      if(last && sub==0) {
        const int64_t loc=p.loc[b];uint8_t* cache=row==H?p.kc:p.vc;
        *(uint16_t*)(cache+(loc*p.HK+kh)*64+2*lane)=kv;
      }
    }
  }
  __syncthreads();proxy_fence();
}
'''

def source():
    s=build.source();a=s.index('template<int L,int QT,int NK>\nDI void prob(')
    s=s[:a]+PREP+'\n'+s[a:]
    s=s.replace('  prep<L,QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);',
        '  if constexpr(L==1)prep_decode<QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);\n  else prep<L,QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split==p.ns-1);')
    ws=build_warp.source();a=ws.index('  auto issue=[&](int tile,int slot) {');b=ws.index('  issue(0,0);',a)
    a0=s.index('  auto issue=[&](int tile,int slot) {');b0=s.index('  issue(0,0);',a0)
    s=s[:a0]+ws[a:b]+s[b0:]
    s=s.replace('constexpr int KB=NK*64,H=8/QT,NR=L*H;', 'constexpr int KB=NK*64,H=8/QT,NR=L*H,NT=128;')
    return s

if __name__=='__main__':
    s=source();p=HERE/'compact_sliding.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
