#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cstdio>
#include <cstdint>
using namespace cute;
int main(){
 auto kl=tile_to_shape(GMMA::Layout_K_SW64_Atom<uint8_t>{},Shape<_64,_64>{});
 auto vl=tile_to_shape(GMMA::Layout_MN_SW128_Atom<uint16_t>{},Shape<_64,_64>{});
 auto pl=tile_to_shape(GMMA::Layout_K_SW128_Atom<uint16_t>{},Shape<_64,_64>{});
 auto k=composition(Swizzle<2,4,3>{},Int<0>{},kl.layout_b());
 auto v=composition(Swizzle<3,3,3>{},Int<0>{},vl.layout_b());
 auto p=composition(Swizzle<3,3,3>{},Int<0>{},pl.layout_b());
 print("K: ");print(k);print("\nV: ");print(v);print("\nP: ");print(p);print("\n");
 for(int r=0;r<64;r++)for(int c=0;c<64;c++){
  int a=r*64+((c>>4)^((r>>1)&3))*16+(c&15);
  int b=c*128+((r>>3)^(c&7))*16+(r&7)*2;
  int d=r*128+((c>>3)^(r&7))*16+(c&7)*2;
  auto aa=k(r,c),bb=2*v(r,c),dd=2*p(r,c);
  if(aa!=a||bb!=b||dd!=d){std::printf("FAIL r%d c%d k %zu/%d v %zu/%d p %zu/%d\n",r,c,size_t(aa),a,size_t(bb),b,size_t(dd),d);return 1;}
 }
 auto uk=recast_layout<uint8_t,uint128_t>(kl.layout_b());
 auto uv=recast_layout<uint16_t,uint128_t>(vl.layout_b());
 auto up=recast_layout<uint16_t,uint128_t>(pl.layout_b());
 print("K canonical u128: ");print(logical_divide(uk,Tile<Layout<_8,_1>,Layout<_2,_1>>{}));print("\n");
 print("V canonical u128: ");print(logical_divide(uv,Tile<Layout<_8,_1>,Layout<_8,_1>>{}));print("\n");
 print("P canonical u128: ");print(logical_divide(up,Tile<Layout<_8,_1>,Layout<_2,_1>>{}));print("\n");
 auto ck=logical_divide(uk,Tile<Layout<_8,_1>,Layout<_2,_1>>{});
 auto cv=logical_divide(uv,Tile<Layout<_8,_1>,Layout<_8,_1>>{});
 auto cp=logical_divide(up,Tile<Layout<_8,_1>,Layout<_2,_1>>{});
 static_assert(stride<0,1>(ck)==32 && stride<1,0>(ck)==1);
 static_assert(stride<0,1>(cv)==0 && stride<1,1>(cv)==64);
 static_assert(stride<0,1>(cp)==64 && stride<1,0>(cp)==1);
 std::puts("PASS descriptor fields: K LBO1/SBO32 B64, V LBO0/SBO64 B128, P LBO1/SBO64 B128");
 std::puts("PASS 4096 coordinates per layout match CUTLASS canonical atoms");
}
