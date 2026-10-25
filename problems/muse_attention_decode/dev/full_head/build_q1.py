"""Single-head persistent full-decode: eight live queries and one cooperative partial-reduction pass."""
import hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_p5_head192',HERE/'build.py');base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
PREP=r'''
DI void prologue(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,int b,int hp,int hh,int seq,int w,int lane,bool virt,float ksc,float vsc) {
  // Each warp owns one real64-dimensional row, unlike64 padded query rows in the verify prologue.
  const int wt=threadIdx.x&127;
  for(int i=wt;i<256;i+=128)if(i*16>=8*64)*(uint4*)(Q+i*16)=make_uint4(0,0,0,0);
  int tk[3],col[3];bool valid[3];float2 x[3];
#pragma unroll
  for(int j=0;j<3;j++) {
    const int row=w+4*j;tk[j]=b;valid[j]=row<8 || (virt && row<10);
    col[j]=row<8?(hp*8+row)*64+2*lane:row==8?(p.HQ+hp)*64+2*lane:row==9?(p.HQ+p.HK+hp)*64+2*lane:0;
  }
  input_pairs<3>(p,tk,col,valid,x);
#pragma unroll
  for(int j=0;j<3;j++) {
    const int row=w+4*j;float2 v=x[j];
    if(row<9) {
      const float n=rsqrtf(wsum(v.x*v.x+v.y*v.y)*(1.f/64.f)+1e-5f);v.x*=n;v.y*=n;
    }
    if(row<8)*(uint16_t*)(Q+off(row,2*lane))=fp8pair(v.x,v.y);
    else if(row<10 && virt) {
      const uint16_t q=fp8pair(v.x/(row==8?ksc:vsc),v.y/(row==8?ksc:vsc));
      if(row==8)*(uint16_t*)(KN+off(0,2*lane))=q;
      else {VN[off(2*lane,0)]=(uint8_t)q;VN[off(2*lane+1,0)]=(uint8_t)(q>>8);}
      const int64_t loc=p.loc[b];uint8_t* cache=row==8?p.kc:p.vc;
      *(uint16_t*)(cache+(loc*p.HK+hp)*64+2*lane)=q;
    }
  }
}

'''
def source():
    s=base.source();a=s.index('DI void prologue(');b=s.index('template <int ST, bool MASK>\nDI void step',a);s=s[:a]+PREP+s[b:]
    old='''    for (int rh = 0; rh < 2; rh++) {
      float lt = l[rh];''';assert old in s
    s=s.replace(old,'''    for (int rh = 0; rh < 2; rh++) {
      if(2*w+rh>=p.L)continue;  // no global stores for padded query rows
      float lt = l[rh];''')
    a=s.index('template <int ST> constexpr int smem_bytes()');s=s[:a].replace('p.L','1').replace('p.window','0')+s[a:]
    old='if (HK % 2 || HQ != 8 * HK || p.NS > NSMAX || (L != 1 && L != 8))';assert old in s
    s=s.replace(old,'if (HQ != 8 * HK || p.NS > NSMAX || L != 1 || window)')
    s=s.replace('* 128 * PR','* 8 * PR').replace('slot * 128 +','slot * 8 +').replace('pb * 128 +','pb * 8 +')
    return s
if __name__=='__main__':
    s=source();p=HERE/'out/full_q1.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
