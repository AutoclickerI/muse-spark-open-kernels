"""Decode sliding: skip padded query rows in FP32 softmax before the native RS-P WGMMA."""
from pathlib import Path
import hashlib,build_last
HERE=Path(__file__).resolve().parent

def source():
    s=build_last.source();a=s.index('DI void prob(');b=s.index('template<int NK> DI void multiply_v',a)
    p=s[a:b];old='''    const int row=warp*16+g+8*rh,tok=row/H;
    const bool live=row<L*H;''';assert old in p
    p=p.replace(old,'''    const int row=warp*16+g+8*rh,tok=row/H;
    if(row>=L*H) {
      alpha[rh]=1.f;
#pragma unroll
      for(int n=0;n<N;n++){s[n][2*rh]=0.f;s[n][2*rh+1]=0.f;}
      continue;
    }
    const unsigned active=__activemask();
    const bool live=true;''').replace('__shfl_xor_sync(0xffffffffu,mx,','__shfl_xor_sync(active,mx,')
    s=s[:a]+p+s[b:]
    a=s.index('DI void virtual_tile(');b=s.index('template<int L,int QT,int NK> constexpr',a)
    p=s[a:b];old='    const int row=w*16+g+rh*8,tok=row/H;';assert old in p
    p=p.replace(old,old+'''\n    if(row>=L*H){alpha[rh]=1.f;continue;}
    const unsigned active=__activemask();''').replace('__shfl_xor_sync(0xffffffffu,mx,','__shfl_xor_sync(active,mx,')
    s=s[:a]+p+s[b:]
    return s

if __name__=='__main__':
    s=source();p=HERE/'out/masked_sliding.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
