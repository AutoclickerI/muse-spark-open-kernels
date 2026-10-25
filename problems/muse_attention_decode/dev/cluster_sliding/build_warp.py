"""Decode-only 32-thread split CTAs: use m16 FP8 MMA instead of padding to native m64 WGMMA."""
from pathlib import Path
import hashlib
import build
HERE=Path(__file__).resolve().parent

def source():
    s='#define USE_WGMMA 0\n'+build.source()
    s=s.replace('for(int i=tid;i<L*32;i+=128)', 'for(int i=tid;i<L*32;i+=blockDim.x)')
    s=s.replace('for(int i=tid;i<4096/16;i+=128)', 'for(int i=tid;i<4096/16;i+=blockDim.x)')
    s=s.replace('  constexpr int KB=NK*64,H=8/QT,NR=L*H;', '  constexpr int KB=NK*64,H=8/QT,NR=L*H,NT=L==1?32:128;')
    s=s.replace('for(int j=0;j<NK/32;j++) {\n      const int row=tid/4+32*j', 'for(int j=0;j<NK/(NT/4);j++) {\n      const int row=tid/4+(NT/4)*j')
    s=s.replace('for(int j=0;j<NK/64;j++)trans64(V+(it%2)*KB+j*4096,VT+j*4096,w,lane);',
        'for(int j=0;j<NK/64;j++)for(int pw=w;pw<4;pw+=NT/32)trans64(V+(it%2)*KB+j*4096,VT+j*4096,pw,lane);')
    s=s.replace('sliding_main<L,QT,NK>,128,cluster_smem', 'sliding_main<L,QT,NK>,L==1?32:128,cluster_smem')
    s=s.replace('cfg.blockDim=dim3(128)', 'cfg.blockDim=dim3(L==1?32:128)')
    s=s.replace('for(int j=w;rank+j*p.cs<NR;j+=4)', 'for(int j=w;rank+j*p.cs<NR;j+=NT/32)')
    a=s.index('  auto issue=[&](int tile,int slot) {');b=s.index('  issue(0,0);',a)
    s=s[:a]+'''  auto issue=[&](int tile,int slot) {
    if(tile<nt) {
      constexpr int R=NK/(NT/4);int pages[R];
#pragma unroll
      for(int j=0;j<R;j++) {
        const int key=begin+tile*NK+tid/4+(NT/4)*j,clamped=min(key,max(seq-1,0));
        pages[j]=p.pt[(int64_t)b*p.pages+(clamped>>p.lg)];
      }
      asm volatile("":::"memory");
#pragma unroll
      for(int j=0;j<R;j++) {
        const int row=tid/4+(NT/4)*j,key=begin+tile*NK+row,clamped=min(key,max(seq-1,0));
        const int physical=pages[j]*p.ps+(clamped&(p.ps-1));
        const int64_t offg=((int64_t)physical*p.HK+kh)*64+16*(tid&3);
        cp16(K+slot*KB+off(row,16*(tid&3)),p.kc+offg,key<end);
        cp16(V+slot*KB+off(row,16*(tid&3)),p.vc+offg,key<end);
      }
    }
    commit();
  };
'''+s[b:]
    assert 'row=tid/4+32*j' not in s
    return s

if __name__=='__main__':
    s=source();p=HERE/'warp_sliding.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
