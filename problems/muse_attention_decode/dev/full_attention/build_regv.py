"""Keep native FP8 PV: V-transpose in register A, P in shared B, O restored once per portion."""
import ast
import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'vq8_base.cu').read_text()
a=s.index('// O[64 x 64 dims]');b=s.index('// N (token, col)',a)
pv=r'''
DI uint8_t* prob_shared(){
 extern __shared__ __align__(1024) uint8_t sm[];
 const int hh=__shfl_sync(0xffffffffu,threadIdx.x>>7,0);
 return sm+CLKPAD+24576+hh*4096;
}
DI float* alpha_shared(){
 extern __shared__ __align__(1024) uint8_t sm[];
 const int hh=__shfl_sync(0xffffffffu,threadIdx.x>>7,0);
 return (float*)(sm+CLKPAD+32768+hh*256);
}
DI void save_alpha(const float(&a)[2]){
 int lane=threadIdx.x&31,w=(threadIdx.x>>5)&3;
 float* p=alpha_shared(); // the full-mask shuffle must execute before the lane predicate
 if((lane&3)==0){p[w*16+(lane>>2)]=a[0];p[w*16+(lane>>2)+8]=a[1];}
}
// P is no longer the register-A operand: after its stores complete, recycle
// those same eight registers for V^T. The caller's protect(pc) after wait keeps
// the native A operands alive without another fragment bank.
DI void pv(float (&o)[8][4],uint32_t (&a)[8],const uint8_t* v){
 int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=(threadIdx.x>>5)&3;
 int hh=__shfl_sync(0xffffffffu,threadIdx.x>>7,0);
 uint8_t* ps=prob_shared();
#pragma unroll
 for(int kk=0;kk<2;++kk)
#pragma unroll
  for(int hf=0;hf<2;++hf)
#pragma unroll
   for(int rh=0;rh<2;++rh){
    int row=w*16+g+8*rh,col=32*kk+16*hf+2*t;
    uint32_t p=a[4*kk+2*hf+rh];
    *(uint16_t*)(ps+off(row,col))=(uint16_t)p;
    *(uint16_t*)(ps+off(row,col+8))=(uint16_t)(p>>16);
   }
 proxy_fence();wg_bar(1+hh);
 // Rescale is deferred from the previous step into this already-required
 // P-publication barrier. Alpha is indexed by O's query columns, not its rows.
 float* al=alpha_shared();
#pragma unroll
 for(int n=0;n<8;++n){
  float x=al[8*n+2*t],y=al[8*n+2*t+1];
  if(__any_sync(0xffffffffu,x!=1.f||y!=1.f)){
   o[n][0]*=x;o[n][2]*=x;o[n][1]*=y;o[n][3]*=y;
  }
 }
#pragma unroll
 for(int kk=0;kk<2;++kk)
#pragma unroll
  for(int hf=0;hf<2;++hf)
#pragma unroll
   for(int rh=0;rh<2;++rh){
    int feature=w*16+g+8*rh,key=32*kk+16*hf+4*t;
    uint32_t z=0;
#pragma unroll
    for(int j=0;j<4;++j)z|=(uint32_t)v[off(key+j,feature)]<<(8*j);
    a[4*kk+2*hf+rh]=z;
   }
 wg_fence();
#if USE_WGMMA
 uint64_t pa=desc(su(ps)),pb=pa+2;pin(pa);pin(pb);
 wg_rs(o,a[0],a[1],a[2],a[3],pa);
 wg_rs(o,a[4],a[5],a[6],a[7],pb);
#else
#pragma unroll
 for(int kk=0;kk<2;++kk)
#pragma unroll
  for(int n=0;n<8;++n)
   mma8(o[n],a[4*kk],a[4*kk+1],a[4*kk+2],a[4*kk+3],word(ps,8*n+g,32*kk+4*t),word(ps,8*n+g,32*kk+16+4*t));
 wg_bar(1+hh); // local warp-MMA shim emulates the native warpgroup's shared-B lifetime
#endif
}
DI void restore_output(float (&o)[8][4],uint8_t* qs,int hh){
 int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=(threadIdx.x>>5)&3;
 uint8_t* tmp=qs+hh*4096;
 auto cell=[&](int i)->float&{return *(float*)(tmp+(i/1024)*8192+(i%1024)*4);};
 // Each head owns one4KiB slice in Q, KN, VN, P. All four are dead at this
 // portion's end. The other head and the producer use disjoint storage.
#pragma unroll
 for(int n=0;n<8;++n)
#pragma unroll
  for(int rh=0;rh<2;++rh)
#pragma unroll
   for(int q=0;q<2;++q)cell((w*16+g+8*rh)*64+8*n+2*t+q)=o[n][2*rh+q];
 wg_bar(1+hh);
#pragma unroll
 for(int n=0;n<8;++n)
#pragma unroll
  for(int rh=0;rh<2;++rh)
#pragma unroll
   for(int q=0;q<2;++q)o[n][2*rh+q]=cell((8*n+2*t+q)*64+w*16+g+8*rh);
 wg_bar(1+hh);
}
'''
s=s[:a]+pv+'\n'+s[b:]
old='for (int i = 0; i < 8; i++) pk[i] = 0u;\n    return;'
assert s.count(old)==1;s=s.replace(old,'for (int i = 0; i < 8; i++) pk[i] = 0u;\n    save_alpha(alpha);\n    return;',1)
old='  // A fragment for PV chunk kk:'
assert s.count(old)==1;s=s.replace(old,'  save_alpha(alpha);\n'+old,1)
a=s.index('DI void rescale(');b=s.index('// A portion',a)
s=s[:a]+'DI void rescale(float (&O)[8][4],const float (&alpha)[2]){} // deferred into pv\n\n'+s[b:]
s=s.replace('      transpose_group(VR + slot * 8192, VT + slot * 8192, pw, lane);','      // Raw V publication only: the consumer forms register-A fragments.')
s=s.replace('uint8_t* Ks = Qs + 24576;','uint8_t* Ks = Qs + 33792;')
s=s.replace('uint8_t* VT = VR + ST * 8192;','uint8_t* VT = VR;')
s=s.replace('CLKPAD + 24576 + ST * 24576','CLKPAD + 33792 + ST * 16384')
old='''    const int pos = 4 * (tok >> 1) + (tok & 1);
    VN[off(2 * lane, pos)] = (uint8_t)(vq & 0xff);
    VN[off(2 * lane + 1, pos)] = (uint8_t)(vq >> 8);'''
assert s.count(old)==1;s=s.replace(old,'    *(uint16_t*)(VN+off(tok,2*lane))=vq;')
old='    const int slot = S.partB[s] + (cta - cta_of(S.segP[s], G, T));'
assert s.count(old)==1;s=s.replace(old,'    restore_output(O,Qs,hh);\n'+old)
ca=s.index('  float2 gate = make_float2',s.index('vq8_combine(VP p)'));cb=s.index("  // This warp's slice",ca)
gate=s[ca:cb];s=s[:ca]+s[cb:]
mark='  *(__nv_bfloat162*)(p.out + ((int64_t)(b * p.L + tok) * p.HQ + head) * 64 + 2 * lane) ='
assert s.count(mark)==1;s=s.replace(mark,gate+mark,1)
# Reuse the independently checked P-fragment fixture, with the new restoration.
probe=(H/'hybrid.cu').read_text().split('__global__ void hybrid_probe',1)[1]
probe='__global__ void hybrid_probe'+probe
probe=probe.replace('__shared__ uint8_t v[4096];','extern __shared__ __align__(1024) uint8_t sm[];uint8_t* v=sm+33792;')
probe=probe.replace(' __syncthreads();uint32_t a[8];', ' if(tid<64)alpha_shared()[tid]=1.f;\n __syncthreads();uint32_t a[8];')
probe=probe.replace(' pv(o,a,v);output_transpose(o);',' wg_fence();pv(o,a,v);wg_commit();wg_wait<0>();protect(o);protect(a);restore_output(o,sm+CLKPAD,0);')
probe=probe.replace('hybrid_probe<<<1,128,0,','hybrid_probe<<<1,128,37888,')
probe=probe.replace('hybrid_probe','regv_probe').replace('hybrid_pv_probe','regv_pv_probe')
s+='\n'+probe
(H/'regv.cu').write_text(s)
tree=ast.parse((H/'vq_runtime.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(s)
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CPP':n.value=ast.Constant(ast.literal_eval(n.value)+'\nvoid regv_pv_probe(int64_t,int64_t,int64_t,int64_t);\n')
drv=ast.unparse(tree).replace("functions=['vq8_forward', 'vq8_resources']","functions=['vq8_forward', 'vq8_resources', 'regv_pv_probe']")+'\n'
(H/'regv.py').write_text(drv)
test=(H/'check_hybrid.py').read_text().replace("load('hybrid')","load('regv')").replace('hybrid_pv_probe','regv_pv_probe').replace("'hybrid.py'","'regv.py'").replace("'hybrid_validation.json'","'regv_validation.json'")
(H/'check_regv.py').write_text(test)
prefix=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
lab=(H/'hybrid_lab.py.inc').read_text().replace('FP8 QK + register FP16 PV','native FP8 PV with V in register A, shared P').replace('V stays raw, exact register widening; P keeps E4M3 quantization, output register transpose only once per portion.','V stays raw, P keeps E4M3 quantization in shared B; alpha is saved by query row and applied by O column at PV publication; output transpose only per portion.')
(H/'regv_lab.py').write_text(prefix+'\nSOURCES='+repr({'old':(H/'vq_runtime.py').read_text(),'new':drv})+'\n'+lab)
print('Native register-V:',hashlib.sha256(s.encode()).hexdigest())
