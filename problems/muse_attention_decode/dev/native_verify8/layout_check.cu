#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cstdint>
#include <cstdio>
#include <cassert>
using namespace cute;
int main(){
 using C=GMMA::CLayout_64x64;
 using K=GMMA::Layout_K_SW64_Atom<uint8_t>;
 for(int t=0;t<128;t++)for(int v=0;v<32;v++){
  int flat=C{}(make_coord(t,v));
  int row=t/32*16+(t%32)/4+(v%4>=2?8:0),col=v/4*8+2*(t%4)+(v%2);
  assert(flat==row+64*col);
 }
 for(int r=0;r<64;r++)for(int c=0;c<64;c++){
  int off=r*64+(((c>>4)^((r>>1)&3))*16)+(c&15);
  assert(off==(r/8)*512+K{}(make_coord(r%8,c)));
 }
 puts("CUTLASS CLayout_64x64: 4096 accumulator coordinates match; Layout_K_SW64: 4096 shared addresses match");
}
