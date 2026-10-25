"""Freeze S14 decode kernels and port the reset softmax arms without editing that queued batch."""
import ast
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3]
base=R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py'
src=base.read_text();assert hashlib.sha256(src.encode()).hexdigest().startswith('f1c77ce6')
tree=ast.parse(src)
def literal(name):return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)
def function(name):
 n=next(n for n in tree.body if getattr(n,'name','')==name);return ''.join(src.splitlines(True)[n.lineno-1:n.end_lineno])
q1=literal('ATTN_PORT_CUDA_SRC');assert hashlib.sha256(q1.encode()).hexdigest().startswith('498e93cc')
reset=R/'problems/muse_bf16_proj/lab/lab_softmax_mix_lab.py'
assert hashlib.sha256(reset.read_bytes()).hexdigest().startswith('03086413')
payload=next(ast.literal_eval(n.value)[0][2] for n in ast.parse(reset.read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_LAB_PAYLOADS')
reset_sources=next(ast.literal_eval(n.value) for n in ast.parse(payload).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SOURCES')
verify_cuda=next(ast.literal_eval(n.value) for n in ast.parse(reset_sources['mix25']).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA')
poly=verify_cuda[verify_cuda.index('DI float sm_poly('):verify_cuda.index('// Online softmax of one 64-key tile;')].replace('DI ','DEVI ')
probe=r'''
__global__ void decode_sum_probe(const float*p,float*out){
 int lane=threadIdx.x&31,g=lane>>2,t=lane&3;float d[4]={};
#pragma unroll
 for(int kk=0;kk<2;++kk){
  uint32_t a[4]={pack_h2(p[g*32+16*kk+2*t],p[g*32+16*kk+2*t+1]),
   pack_h2(p[(g+8)*32+16*kk+2*t],p[(g+8)*32+16*kk+2*t+1]),
   pack_h2(p[g*32+16*kk+8+2*t],p[g*32+16*kk+9+2*t]),
   pack_h2(p[(g+8)*32+16*kk+8+2*t],p[(g+8)*32+16*kk+9+2*t])};
  mma16816(d,a,0x3c003c00u,0x3c003c00u);
 }
 out[g*8+2*t]=d[0];out[g*8+2*t+1]=d[1];out[(g+8)*8+2*t]=d[2];out[(g+8)*8+2*t+1]=d[3];
}
__global__ void decode_exp_probe(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=sm_poly(x[i]);}
void decode_probes(int64_t p,int64_t sums,int64_t x,int64_t y,int64_t n,int64_t stream){
 decode_sum_probe<<<1,32,0,(cudaStream_t)stream>>>((const float*)p,(float*)sums);
 decode_exp_probe<<<(n+127)/128,128,0,(cudaStream_t)stream>>>((const float*)x,(float*)y,n);
 auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
'''
clock=r'''
#ifndef DECODE_CLOCK
#define DECODE_CLOCK 0
#endif
#if DECODE_CLOCK
#define DMARK(t) const unsigned long long t=clock64()
#define DADD(i,t) if((threadIdx.x&31)==0)fc[i]+=clock64()-t
__device__ __forceinline__ unsigned long long dtime(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
struct DecodeTimer {
 unsigned long long*s,*d;
 __device__ DecodeTimer(unsigned long long*ss,unsigned long long*dd):s(ss),d(dd){if((threadIdx.x&31)==0){for(int i=0;i<32;++i)s[i]=0;s[0]=dtime();s[1]=clock64();}}
 __device__ ~DecodeTimer(){if((threadIdx.x&31)==0){s[12]=clock64()-s[1];s[13]=dtime();s[9]=s[14]?clock64()-s[14]:0;for(int i=0;i<32;++i)d[i]=s[i];}}
};
#else
#define DMARK(t)
#define DADD(i,t)
#endif
'''
def source(mix=0,tc=False):
 s=q1.replace('namespace cg =',clock+'\n'+poly+'\nnamespace cg =',1)
 # poly helper is before DEVI in the original, so spell its attribute explicitly.
 s=s.replace('DEVI float sm_poly','__device__ __forceinline__ float sm_poly')
 a=s.index('  // Online softmax per row');b=s.index('  // O += P V.',a);soft=s[a:b]
 old='        const float p0 = ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use));'
 assert soft.count(old)==1
 if mix:
  exp='sm_poly(fmaf(s[mt][n][2 * rh], scale_log2, -m_use))'
  if mix==25:exp='((n&1)==0 ? '+exp+' : ex2(fmaf(s[mt][n][2 * rh], scale_log2, -m_use)))'
  soft=soft.replace(old,'        float p0 = '+exp+';').replace('        const float p1 =','        float p1 =')
  soft=soft.replace('        s[mt][n][2 * rh] = p0;','        asm volatile("":"+f"(p0),"+f"(p1)::"memory");\n        s[mt][n][2 * rh] = p0;')
 if tc:
  soft=soft.replace('        sum += p0 + p1;','        // P row sums use the exact FP16 P operands below.')
  soft=soft.replace('      st.l[mt][rh] += sum;','')
  end=soft.rfind('\n  }')
  soft=soft[:end]+'''
    float den[4]={0.f,0.f,0.f,0.f};
#pragma unroll
    for(int kk=0;kk<KK;++kk)mma16816(den,pa[mt][kk],0x3c003c00u,0x3c003c00u);
    // MMA columns all contain the full row sum; final four-lane sum is unchanged.
    st.l[mt][0]+=den[0]*.25f;
    if constexpr(L!=1)st.l[mt][1]+=den[2]*.25f;
'''+soft[end:]
 soft=soft.replace('      float mx = -INFINITY;','      DMARK(tmax);\n      float mx = -INFINITY;')
 soft=soft.replace('      const float m_new =','      DADD(15,tmax);DMARK(talpha);\n      const float m_new =')
 soft=soft.replace('      float sum = 0.f;','      DADD(16,talpha);DMARK(texp);\n      float sum = 0.f;')
 soft=soft.replace('    }\n#pragma unroll\n    for (int kk = 0; kk < KK;', '      DADD(17,texp);\n    }\n    DMARK(tpack);\n#pragma unroll\n    for (int kk = 0; kk < KK;',1)
 # Count exact rescale opportunities without changing the incumbent's unconditional multiply.
 soft=soft.replace('      const float corr = ex2(st.m[mt][rh] - m_use);','''      const float corr = ex2(st.m[mt][rh] - m_use);
#if DECODE_CLOCK
      if((threadIdx.x&31)==0){fc[20]++;fc[21]+=(corr!=1.f);}
#endif''')
 end=soft.rfind('\n  }');soft=soft[:end]+'\n    DADD(18,tpack);'+soft[end:]
 s=s[:a]+soft+s[b:]
 s=s.replace('bool need_mask, float scale_log2) {','bool need_mask, float scale_log2,unsigned long long*fc=nullptr) {\n  DMARK(tqk);',1)
 s=s.replace('  // Online softmax per row','  DADD(6,tqk);DMARK(tsmx);\n  // Online softmax per row',1)
 s=s.replace('  // O += P V.','  DADD(7,tsmx);DMARK(tpv);\n  // O += P V.',1)
 mark='\n}\n\ntemplate <int L, int STAGES>\n__global__';assert s.count(mark)==1
 s=s.replace(mark,'\n  DADD(8,tpv);'+mark,1)
 mark='  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;'
 assert s.count(mark)==1
 s=s.replace(mark,mark+'''
#if DECODE_CLOCK
  __shared__ unsigned long long timers[4][32];unsigned long long*fc=timers[warp];
  const long long nr=(long long)p.batch*(p.hkv/2)*(p.ns/p.cs)*L*16;
  auto dst=reinterpret_cast<unsigned long long*>(p.part_ml+nr*2)+(((b*(p.hkv/2)+kvg)*p.ns+split)*4+warp)*32;
  DecodeTimer timer(fc,dst);DMARK(tprep);
#else
  unsigned long long*fc=nullptr;
#endif
''',1)
 old='  for (int i = 0; i < ntiles; ++i) {\n    cp_wait<STAGES - 2>();\n    __syncthreads();'
 new='''  DADD(2,tprep);DMARK(tloop);
  for (int i = 0; i < ntiles; ++i) {
    DMARK(twait);cp_wait<STAGES - 2>();DADD(3,twait);DMARK(tsync);
    __syncthreads();DADD(4,tsync);DMARK(tissue);'''
 assert s.count(old)==1;s=s.replace(old,new)
 s=s.replace('    load_rows(nxt + 1, rows_next);','    load_rows(nxt + 1, rows_next);\n    DADD(5,tissue);',1)
 s=s.replace('need_mask, scale_log2);','need_mask, scale_log2,fc);',1).replace('true, scale_log2);','true, scale_log2,fc);',1)
 mark='  cp_wait<0>();\n  if (last &&';assert s.count(mark)==1
 s=s.replace(mark,'''  DADD(11,tloop);
#if DECODE_CLOCK
  if(lane==0){fc[10]=ntiles;fc[14]=clock64();}
#endif
  cp_wait<0>();
  if (last &&''',1)
 s+=probe
 return s

common='''import hashlib
import torch
from torch.utils.cpp_extension import load_inline
_EXT={}
_PLANS={}
_COUNTERS={}
'''
cpp=literal('ATTN_PORT_CPP_SRC')+'\nvoid decode_probes(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
manifest=[]
for name,mix,tc in [('ref',0,False),('mix25',25,False),('mix50',50,False),('tc',0,True),('tc25',25,True)]:
 cu=source(mix,tc);(H/(name+'.cu')).write_text(cu)
 drv=common+'CUDA='+repr(cu)+'\nCPP='+repr(cpp)+'\n'+function('_attn_main_plan')+'\n'+(H/'driver.py.inc').read_text()
 (H/(name+'.py')).write_text(drv);manifest.append(dict(name=name,kind='attn_main',mix=mix,tc=tc,cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),driver_sha256=hashlib.sha256(drv.encode()).hexdigest()))
for old,name in [('ref','wgref'),('mix25','wg25'),('mix50','wg50'),('ping','ping'),('ping25','ping25')]:
 drv=ast.parse(reset_sources[old])
 cu=next(ast.literal_eval(n.value) for n in drv.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA')
 if old.startswith('ping'):
  # q_len1: only the first warp has live query rows; no TC operation is in this branch.
  cu=cu.replace('pn,sc,kbase,lo,hi,true)','pn,sc,kbase,lo,hi,2*((threadIdx.x>>5)&3)<p.L)')
  assert 'pn,sc,kbase,lo,hi,true)' not in cu
 for n in drv.body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(cu)
 text=ast.unparse(drv)+'\n'+(H/'wg_driver.py.inc').read_text()
 (H/(name+'.cu')).write_text(cu);(H/(name+'.py')).write_text(text)
 manifest.append(dict(name=name,kind='two_WG',reset_arm=old,cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),driver_sha256=hashlib.sha256(text.encode()).hexdigest()))
(H/'manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(src.encode()).hexdigest(),reset_unchanged_sha256=hashlib.sha256(reset.read_bytes()).hexdigest(),variants=manifest),indent=2)+'\n')
print([(x['name'],x['cuda_sha256'][:8]) for x in manifest])
