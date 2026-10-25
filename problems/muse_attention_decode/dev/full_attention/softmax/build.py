"""S14 softmax issue study: split SFU/FMA exponentials, tensor-core sums and WG staggering."""
import ast
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3]
base=R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py'
s=base.read_text();assert hashlib.sha256(s.encode()).hexdigest().startswith('f1c77ce6')
tree=ast.parse(s)
cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA')
assert hashlib.sha256(cuda.encode()).hexdigest().startswith('c262e4c5')
poly=r'''
// FP32 nearest-integer Cody-Waite reduction; fractional domain[-.5,.5].
// Our existing degree3 minimax, not the old all-polynomial schedule.
// The magic add avoids F2I/I2F. The low bits encode signed n modulo2^9.
DI float sm_poly(float x){
 uint32_t out;
 // Keep temporaries local to this expression while PV operands are pending.
 asm volatile("{.reg .f32 xc,r,n,f,p; .reg .b32 e,b;\n"
  "max.f32 xc,%1,0fc2fc0000;\n"
  "add.rn.f32 r,xc,0f4b400000;\n"
  "sub.rn.f32 n,r,0f4b400000;sub.rn.f32 f,xc,n;\n"
  "fma.rn.f32 p,f,0f3d64ddb8,0f3e781c0b;\n"
  "fma.rn.f32 p,p,f,0f3f3170ca;fma.rn.f32 p,p,f,0f3f7ffc9c;\n"
  "mov.b32 e,r;shl.b32 e,e,23;mov.b32 b,p;add.s32 %0,e,b;}"
  :"=r"(out):"f"(x):"memory");
 return __uint_as_float(out);
}
'''
sumhelper=r'''
DI void sm_tc_sum(float (&l)[2],const uint32_t (&p)[8]){
 extern __shared__ __align__(1024) uint8_t sm[];
 float d[4]={0.f,0.f,0.f,0.f};
#if USE_WGMMA
 uint64_t b=desc(su(sm+CLKPAD+24576));pin(b);
 wg_fence();
#pragma unroll
 for(int j=0;j<2;++j){
  uint64_t bj=b+2*j;pin(bj);
  asm volatile("{.reg .pred q;setp.ne.b32 q,1,0;wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},%8,q,1,1;}"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(p[4*j]),"r"(p[4*j+1]),"r"(p[4*j+2]),"r"(p[4*j+3]),"l"(bj));
 }
 wg_commit();wg_wait<0>();
 for(int i=0;i<4;++i)asm volatile("":"+f"(d[i])::"memory");
 for(int i=0;i<8;++i)asm volatile(""::"r"(p[i]):"memory");
#else
 // Exact FP8-P sum stand-in: Hopper-only M64N8 RS WGMMA above is compiled separately.
 for(int j=0;j<8;++j){
  float2 a=__half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(p[j]&65535),__NV_E4M3));
  float2 b=__half22float2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(p[j]>>16),__NV_E4M3));
  d[j&1]+=a.x+a.y+b.x+b.y;
 }
 d[0]=__shfl_xor_sync(0xffffffffu,d[0],1)+d[0];d[0]=__shfl_xor_sync(0xffffffffu,d[0],2)+d[0];
 d[2]=__shfl_xor_sync(0xffffffffu,d[1],1)+d[1];d[2]=__shfl_xor_sync(0xffffffffu,d[2],2)+d[2];
#endif
 // Each of four lanes carries one quarter; established final reduction is unchanged.
 l[0]+=d[0]*.25f;l[1]+=d[2]*.25f;
}
'''
probe=r'''
__global__ void sm_probe(const float* x,const float* ml,uint32_t* packed,float* stats,float* oo,float scale){
 extern __shared__ __align__(1024) uint8_t sm[];
 int tid=threadIdx.x,lane=tid&31,g=lane>>2,t=lane&3,w=tid>>5;
 if(tid<32)*(uint4*)(sm+CLKPAD+24576+16*tid)=make_uint4(0x38383838,0x38383838,0x38383838,0x38383838);
 proxy_fence();__syncthreads();
 float sx[8][4],m[2],l[2],alpha[2],o[8][4];uint32_t pk[8];
 for(int rh=0;rh<2;++rh){int row=16*w+g+8*rh;m[rh]=ml[2*row];l[rh]=ml[2*row+1];}
 for(int n=0;n<8;++n)for(int rh=0;rh<2;++rh)for(int e=0;e<2;++e){sx[n][2*rh+e]=x[(w*16+g+8*rh)*64+8*n+2*t+e];o[n][2*rh+e]=sx[n][2*rh+e];}
 const int lo[2]={-1000,-1000},hi[2]={1000,1000};
 softmax<false>(sx,m,l,alpha,pk,scale,0,lo,hi,true);rescale(o,alpha);
 for(int i=0;i<8;++i)packed[tid*8+i]=pk[i];
 for(int rh=0;rh<2;++rh){stats[tid*6+rh]=m[rh];stats[tid*6+2+rh]=l[rh];stats[tid*6+4+rh]=alpha[rh];}
 for(int n=0;n<8;++n)for(int e=0;e<4;++e)oo[tid*32+n*4+e]=o[n][e];
}
__global__ void sm_exp_probe(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=sm_poly(x[i]);}
void softmax_probe(int64_t x,int64_t ml,int64_t p,int64_t stats,int64_t o,double scale,int64_t stream){
 sm_probe<<<1,128,CLKPAD+25600,(cudaStream_t)stream>>>((const float*)x,(const float*)ml,(uint32_t*)p,(float*)stats,(float*)o,(float)scale);
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void softmax_exp_probe(int64_t x,int64_t y,int64_t n,int64_t stream){sm_exp_probe<<<(n+127)/128,128,0,(cudaStream_t)stream>>>((const float*)x,(float*)y,(int)n);}
'''
def source(mix=0,gap=0,tc=False):
 s=cuda
 # Same combine math; avoid the incumbent's unrelated gate-load liveness spill.
 ca=s.index('  float2 gate = make_float2',s.index('vq8_combine(VP p)'));cb=s.index("  // This warp's slice",ca)
 gate=s[ca:cb];s=s[:ca]+s[cb:]
 mark='  *(__nv_bfloat162*)(p.out + ((int64_t)(b * p.L + tok) * p.HQ + head) * 64 + 2 * lane) ='
 s=s.replace(mark,gate+mark,1)
 s=s.replace('// Online softmax of one 64-key tile;',poly+(sumhelper if tc else '')+'\n// Online softmax of one 64-key tile;',1)
 a=s.index('DI void softmax(');b=s.index('DI void rescale(',a);sm=s[a:b]
 sm=sm.replace('  const int t = threadIdx.x & 3;','  CLK_MARK(smax);\n  const int t = threadIdx.x & 3;')
 mark='  for (int rh = 0; rh < 2; rh++) {\n    const float mn'
 assert sm.count(mark)==1
 sm=sm.replace(mark,'  for (int rh = 0; rh < 2; rh++) {\n    CLK_ADD(0,64,smax);CLK_MARK(sa);\n    const float mn')
 sm=sm.replace('    float sum0 = 0.f, sum1 = 0.f;','    CLK_ADD(0,65,sa);CLK_MARK(se);\n    float sum0 = 0.f, sum1 = 0.f;')
 old='      const float p0 = ex2(fmaf(s[n][2 * rh], sc, bias)), p1 = ex2(fmaf(s[n][2 * rh + 1], sc, bias));'
 assert sm.count(old)==1
 if mix:
  pp='sm_poly(fmaf(s[n][2 * rh], sc, bias))'
  if mix==25:pp='((n&1)==0 ? '+pp+' : ex2(fmaf(s[n][2 * rh], sc, bias)))'
  sm=sm.replace(old,'      const float p0 = '+pp+', p1 = ex2(fmaf(s[n][2 * rh + 1], sc, bias));')
  sm=sm.replace('      const float p0 =','      float p0 =').replace('      s[n][2 * rh] = p0;','      asm volatile("":"+f"(p0),"+f"(p1)::"memory");\n      s[n][2 * rh] = p0;')
 sm=sm.replace('    l[rh] = l[rh] * alpha[rh] + (sum0 + sum1);','    CLK_ADD(0,66,se);CLK_MARK(sl);\n    l[rh] = l[rh] * alpha[rh]'+(';' if tc else ' + (sum0 + sum1);')+'\n    CLK_ADD(0,67,sl);')
 if tc:sm=sm.replace('      sum0 += p0;\n      sum1 += p1;','      // Row sum is native FP8 WGMMA(P,ones), after P packing.')
 sm=sm.replace('  // A fragment for PV chunk kk:','  CLK_MARK(sp);\n  // A fragment for PV chunk kk:')
 end=sm.rfind('\n}')
 sm=sm[:end]+'\n  CLK_ADD(0,68,sp);'+('\n  sm_tc_sum(l,pk);' if tc else '')+sm[end:]
 # Do not count the prefix max timer twice for the second row.
 sm=sm.replace('    CLK_ADD(0,64,smax);CLK_MARK(sa);','    if(rh==0){CLK_ADD(0,64,smax);}\n    CLK_MARK(sa);')
 s=s[:a]+sm+s[b:]
 mark='DI void rescale(float (&O)[8][4], const float (&alpha)[2]) {'
 s=s.replace(mark,mark+'\n  CLK_MARK(sr);\n#if VQ8_CLK\n  if(threadIdx.x==0)clk_area()[69]++;\n#endif')
 mark='  if (__any_sync(0xffffffffu, alpha[0] != 1.f || alpha[1] != 1.f)) {'
 s=s.replace(mark,mark+'\n#if VQ8_CLK\n    if(threadIdx.x==0)clk_area()[70]++;\n#endif')
 end=s.index('\n}\n\n// A portion',s.index('DI void rescale'))
 s=s[:end]+'\n  CLK_ADD(0,71,sr);'+s[end:]
 s=s.replace('if (tid < 64) clk_area()[tid] = 0;','if (tid < 128) clk_area()[tid] = 0;')
 mark='    row[19] = a[48]; row[20] = a[49]; row[21] = a[50]; row[22] = a[51];'
 s=s.replace(mark,mark+'\n    row[25]=a[64];row[26]=a[66];row[27]=a[68];row[28]=a[65]+a[67];row[29]=a[69];row[30]=a[70];row[31]=a[71];')
 if gap:
  mark='    float O[8][4], m[2]'
  assert s.count(mark)==1
  s=s.replace(mark,f'''    // One initial WG stagger, not a per-tile delay or forced turn barrier.
    if(hh==1 && po==0){{unsigned long long t=clock64();while(clock64()-t<{gap}ull)asm volatile("":::"memory");}}
'''+mark)
 if tc:
  s=s.replace('uint8_t* Ks = Qs + 24576;','uint8_t* Ks = Qs + 25600;')
  s=s.replace('CLKPAD + 24576 + ST * 24576','CLKPAD + 25600 + ST * 24576')
  mark='  if (tid < ST) { mb_init(&full[tid], 4); mb_init(&empty[tid], 8); }'
  assert s.count(mark)==1
  s=s.replace(mark,'  if(tid<32)*(uint4*)(sm+CLKPAD+24576+16*tid)=make_uint4(0x38383838,0x38383838,0x38383838,0x38383838);\n  proxy_fence();\n'+mark)
 return s+'\n'+probe

specs={'ref':(0,0,False),'mix25':(25,0,False),'mix50':(50,0,False),
       'gap':(0,800,False),'mixgap':(25,800,False),'tc25':(25,0,True)}
manifest=[]
for name,(mix,gap,tc) in specs.items():
 cu=source(mix,gap,tc);(H/(name+'.cu')).write_text(cu)
 drv=ast.parse((P/'vq_runtime.py').read_text())
 for n in drv.body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(cu)
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CPP':n.value=ast.Constant(ast.literal_eval(n.value)+'\nvoid softmax_probe(int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t);\nvoid softmax_exp_probe(int64_t,int64_t,int64_t,int64_t);\n')
 text=ast.unparse(drv).replace("functions=['vq8_forward', 'vq8_resources']","functions=['vq8_forward', 'vq8_resources','softmax_probe','softmax_exp_probe']")+'\n'
 (H/(name+'.py')).write_text(text)
 manifest.append(dict(name=name,mix=mix,gap=gap,tc=tc,cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),driver_sha256=hashlib.sha256(text.encode()).hexdigest()))
(H/'manifest.json').write_text(json.dumps(dict(base=str(base.relative_to(R)),base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),variants=manifest),indent=2)+'\n')
print([(x['name'],x['cuda_sha256'][:8]) for x in manifest])
