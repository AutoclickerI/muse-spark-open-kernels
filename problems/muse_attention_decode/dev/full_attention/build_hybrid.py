"""Native FP8 QK, register P, exact FP8-to-FP16 widening for PV; no shared V transpose."""
import ast
import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'vq8_base.cu').read_text()
a=s.index('// O[64 x 64 dims]');b=s.index('// N (token, col)',a)
pv=r'''
DI uint32_t h2_from_fp8(uint32_t x){
 uint32_t y;uint16_t lo=(uint16_t)x;
 asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(y):"h"(lo));return y;
}
DI void hmma(float (&o)[4],const uint32_t(&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
 :"+f"(o[0]),"+f"(o[1]),"+f"(o[2]),"+f"(o[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// Accumulator dimension order is temporarily transposed: 8*(2t+q)+n.
// Vector raw-V loads are reused over eight dimension tiles, as in attn_main.
DI void pv(float (&o)[8][4],const uint32_t(&p)[8],const uint8_t* v){
 const int lane=threadIdx.x&31,g=lane>>2,t=lane&3;
#pragma unroll
 for(int kk=0;kk<4;++kk){
  uint32_t a[4]={h2_from_fp8(p[2*kk]),h2_from_fp8(p[2*kk+1]),h2_from_fp8(p[2*kk]>>16),h2_from_fp8(p[2*kk+1]>>16)};
  const int r=16*kk+2*t;
  uint2 va=*(const uint2*)(v+off(r,8*g)),vb=*(const uint2*)(v+off(r+1,8*g));
  uint2 vc=*(const uint2*)(v+off(r+8,8*g)),vd=*(const uint2*)(v+off(r+9,8*g));
#pragma unroll
  for(int j=0;j<2;++j){
   uint32_t aa=j?va.y:va.x,bb=j?vb.y:vb.x,cc=j?vc.y:vc.x,dd=j?vd.y:vd.x;
   uint32_t ab0=prmt(aa,bb,0x5140),ab1=prmt(aa,bb,0x7362),cd0=prmt(cc,dd,0x5140),cd1=prmt(cc,dd,0x7362);
   uint32_t b0[4]={h2_from_fp8(ab0),h2_from_fp8(ab0>>16),h2_from_fp8(ab1),h2_from_fp8(ab1>>16)};
   uint32_t b1[4]={h2_from_fp8(cd0),h2_from_fp8(cd0>>16),h2_from_fp8(cd1),h2_from_fp8(cd1>>16)};
#pragma unroll
   for(int z=0;z<4;++z)hmma(o[4*j+z],a,b0[z],b1[z]);
  }
 }
}
DI void output_transpose(float (&o)[8][4]){
 const int t=threadIdx.x&3;
 // Swap the low dimension bit in the register index with the local pair bit.
#pragma unroll
 for(int n=0;n<8;n+=2)
#pragma unroll
  for(int rh=0;rh<2;++rh){float x=o[n][2*rh+1];o[n][2*rh+1]=o[n+1][2*rh];o[n+1][2*rh]=x;}
 // Swap the remaining two dimension bits with the quad's lane bits.
#pragma unroll
 for(int bit=1;bit<=2;++bit){
#pragma unroll
  for(int n=0;n<8;++n)if(!(n&(1<<bit))){
#pragma unroll
   for(int e=0;e<4;++e){
    float x=__shfl_xor_sync(0xffffffffu,o[n+(1<<bit)][e],1<<(bit-1));
    float y=__shfl_xor_sync(0xffffffffu,o[n][e],1<<(bit-1));
    if(t&(1<<(bit-1)))o[n][e]=x;else o[n+(1<<bit)][e]=y;
   }
  }
 }
}
'''
s=s[:a]+pv+'\n'+s[b:]
old='      transpose_group(VR + slot * 8192, VT + slot * 8192, pw, lane);'
assert s.count(old)==1;s=s.replace(old,'      // Raw V is published directly; consumer widens exactly in registers.')
s=s.replace('uint8_t* VT = VR + ST * 8192;','uint8_t* VT = VR;')
s=s.replace('CLKPAD + 24576 + ST * 24576','CLKPAD + 24576 + ST * 16384')
old='''    const int pos = 4 * (tok >> 1) + (tok & 1);
    VN[off(2 * lane, pos)] = (uint8_t)(vq & 0xff);
    VN[off(2 * lane + 1, pos)] = (uint8_t)(vq >> 8);'''
assert s.count(old)==1;s=s.replace(old,'    *(uint16_t*)(VN+off(tok,2*lane))=vq;')
s=s.replace('  wg_wait<1>();','  wg_wait<0>();')
# Keep the two gate values out of the split-reduction loop. The original
# 64-register combine spills with this image/compiler; this preserves all math.
ca=s.index('  float2 gate = make_float2',s.index('vq8_combine(VP p)'))
cb=s.index("  // This warp's slice",ca)
gate=s[ca:cb];s=s[:ca]+s[cb:]
mark='  *(__nv_bfloat162*)(p.out + ((int64_t)(b * p.L + tok) * p.HQ + head) * 64 + 2 * lane) ='
assert s.count(mark)==1;s=s.replace(mark,gate+mark,1)
old='    const int slot = S.partB[s] + (cta - cta_of(S.segP[s], G, T));'
assert s.count(old)==1;s=s.replace(old,'    output_transpose(O);\n'+old)
# A standalone arithmetic/layout probe, independent of the attention error tolerance.
s+=r'''
__global__ void hybrid_probe(const uint8_t* P,const uint8_t* V,float* out){
 __shared__ uint8_t v[4096];int tid=threadIdx.x,lane=tid&31,g=lane>>2,t=lane&3,w=tid>>5;
 for(int i=tid;i<4096;i+=128)v[off(i/64,i%64)]=V[i];
 __syncthreads();uint32_t a[8];float o[8][4]={};
#pragma unroll
 for(int kk=0;kk<2;++kk)
#pragma unroll
  for(int hf=0;hf<2;++hf){
   int n=4*kk+2*hf;
   a[4*kk+2*hf]=(uint32_t)*(const uint16_t*)(P+(w*16+g)*64+8*n+2*t)|((uint32_t)*(const uint16_t*)(P+(w*16+g)*64+8*(n+1)+2*t)<<16);
   a[4*kk+2*hf+1]=(uint32_t)*(const uint16_t*)(P+(w*16+g+8)*64+8*n+2*t)|((uint32_t)*(const uint16_t*)(P+(w*16+g+8)*64+8*(n+1)+2*t)<<16);
  }
 pv(o,a,v);output_transpose(o);
#pragma unroll
 for(int n=0;n<8;++n)
#pragma unroll
  for(int rh=0;rh<2;++rh)
#pragma unroll
   for(int q=0;q<2;++q)out[(w*16+g+rh*8)*64+8*n+2*t+q]=o[n][2*rh+q];
}
void hybrid_pv_probe(int64_t p,int64_t v,int64_t o,int64_t stream){
 hybrid_probe<<<1,128,0,(cudaStream_t)stream>>>((const uint8_t*)p,(const uint8_t*)v,(float*)o);
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
'''
(H/'hybrid.cu').write_text(s)
drv=(H/'vq_runtime.py').read_text();tree=ast.parse(drv)
for n in tree.body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(s)
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CPP':n.value=ast.Constant(ast.literal_eval(n.value)+'\nvoid hybrid_pv_probe(int64_t,int64_t,int64_t,int64_t);\n')
drv=ast.unparse(tree).replace("functions=['vq8_forward', 'vq8_resources']","functions=['vq8_forward', 'vq8_resources', 'hybrid_pv_probe']")+'\n'
(H/'hybrid.py').write_text(drv)
prefix=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
(H/'hybrid_lab.py').write_text(prefix+'\nSOURCES='+repr({'old':(H/'vq_runtime.py').read_text(),'new':drv})+'\n'+(H/'hybrid_lab.py.inc').read_text())
print('Hybrid FP8 QK / exact FP16 PV:',hashlib.sha256(s.encode()).hexdigest())
