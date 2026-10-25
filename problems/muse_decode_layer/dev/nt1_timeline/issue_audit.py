"""Compile marked SASS probes of the actual NT1 math/epilogue bodies; model issue separately from latency."""
import ast,collections,hashlib,json,re,subprocess
from pathlib import Path
import build_reuse as reuse
from build_epilogue import VEC
from build_epilogue_pair import PAIR
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
s=ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py'
t=ast.parse(s.read_text());cuda=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SMALLM_VAR_CUDA')
head=cuda[:cuda.index('enum {\n  F_Z')]
head+='\n__device__ __forceinline__ uint64_t mark(){uint64_t x;asm volatile("mov.u64 %0,%%clock64;":"=l"(x)::"memory");return x;}\n'
math=r'''
template<int NB>__global__ __launch_bounds__(256,2) void p_math(const uint4* wg,const uint4* sg,const __nv_bfloat16* xg,const int* pg,float* output,uint64_t* stamps){
 __shared__ uint4 qw[8*32],sc[32];__shared__ int routepos[32];int lane=threadIdx.x&31,warp=0,tid=lane,pass=0,ne=32,off=0;constexpr int HD=4096;
 for(int i=lane;i<8*32;i+=32)qw[i]=wg[i];sc[lane]=sg[lane];routepos[lane]=pg[lane];__syncthreads();
 const int16_t* pair_of=nullptr;const __nv_bfloat16* inter=xg;const __nv_bfloat16* x=xg;
 uint32_t cached_b[8][8];const uint32_t* cached_last=nullptr;
 uint64_t b0=mark();
#pragma unroll
 for(int g=0;g<8;++g){const uint4* p=reinterpret_cast<const uint4*>(x+(size_t)(routepos[lane>>2]>>3)*4096+g*64+(lane&3)*16);
  uint4 a=__ldg(p),b=__ldg(p+1);cached_b[g][0]=a.x;cached_b[g][1]=a.y;cached_b[g][2]=a.z;cached_b[g][3]=a.w;cached_b[g][4]=b.x;cached_b[g][5]=b.y;cached_b[g][6]=b.z;cached_b[g][7]=b.w;}
 uint64_t b1=mark();
 float acc[NB][4]={};const uint4* wq=qw+lane;const uint4 scv=sc[lane];const uint32_t scw[4]={scv.x,scv.y,scv.z,scv.w};
 uint64_t m0=mark();
#pragma unroll
 for(int g=0;g<8;++g){
// DECODE
#pragma unroll
 for(int j=0;j<NB;++j){
// MMA
 }
 }
 uint64_t m1=mark();
#pragma unroll
 for(int j=0;j<NB;++j)*reinterpret_cast<float4*>(output+(j*32+lane)*4)=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
 stamps[lane*4]=b0;stamps[lane*4+1]=b1;stamps[lane*4+2]=m0;stamps[lane*4+3]=m1;
}
'''
decode=reuse.DECODE
mma=reuse.MMA.replace('pair_of[pos]','routepos[pos]').replace('g<7?cached_b[g][s*2]:cached_last[(s*2)*256+tid]', 'cached_b[g][s*2]').replace('g<7?cached_b[g][s*2+1]:cached_last[(s*2+1)*256+tid]', 'cached_b[g][s*2+1]')
math=math.replace('// DECODE',decode).replace('// MMA',mma)
a=cuda.index('    if (pass == 0) {',cuda.index('    const float* rb = red + buf'))
b=cuda.index('    if constexpr (TRACE) {\n      if (tid == 0 && pass == 0 && firstp)',a)
epi=cuda[a:b]
# Fold the role and pass, retaining actual scalar shared loads, sum order, activation/norm work and stores.
epi_header=r'''
template<int PASS,int ROLE,bool VECTOR>__global__ void p_epi(const float* input,const __nv_bfloat16* pw,__nv_bfloat16* inter,float* y,float* ss,uint64_t* stamps,int m){
 __shared__ float red[8*32*4];int lane=threadIdx.x;constexpr int warp=ROLE,pass=PASS,NT=1,NB=8,RED=4;constexpr bool Z=true,ZOUT=true,NREG=false,VNR=false;
 int tid=ROLE*32+lane,nb=0,ne=8,off=0,e=0;const __nv_bfloat16* pnw=pw;float fw[2],nacc=0.f;int npos=-1;float nv0=0,nv1=0;
 for(int i=lane;i<8*32*4;i+=32)red[i]=input[i];__syncthreads();const float* rb=red;
 const float* gs13=input+1024;const float* gs2=input+1056;
 uint64_t t0=mark();
 if constexpr(!VECTOR&&PASS==1)fw[0]=1.f+__bfloat162float(pnw[m*16+(tid&15)]);
 if constexpr(VECTOR){
// VEC
 }else{
// EPI
 }
 uint64_t t1=mark();stamps[lane*2]=t0;stamps[lane*2+1]=t1;
}
'''
epi_header=epi_header.replace('// VEC',VEC).replace('// EPI',epi)
extra=r'''
__global__ void p_store(const float4* p,float* output,uint64_t* stamps){__shared__ float r[1024];int l=threadIdx.x;float4 a=p[l];uint64_t t0=mark();
 *reinterpret_cast<float4*>(r+l*4)=a;__syncthreads();uint64_t t1=mark();*reinterpret_cast<float4*>(output+l*4)=*reinterpret_cast<float4*>(r+l*4);stamps[l*2]=t0;stamps[l*2+1]=t1;}
__global__ void p_prefetch(const Route* input,const uint8_t* w13,const uint8_t* w2,uint64_t* stamps,int total,int first,int a2){
 __shared__ Route r;__shared__ __align__(128) uint8_t stage[2*TILE_BYTES];__shared__ uint64_t full[2],empty[2];int l=threadIdx.x;
 for(int i=l;i<(int)sizeof(Route)/4;i+=32)reinterpret_cast<int*>(&r)[i]=reinterpret_cast<const int*>(input)[i];__syncthreads();
 uint64_t t0=mark();if(l==0)prefetch_one(&r,total,first,a2,2,stage,full,empty,w13,w2);uint64_t t1=mark();stamps[l*2]=t0;stamps[l*2+1]=t1;
}
'''
inst='\n'.join(f'template __global__ void p_math<{nb}>(const uint4*,const uint4*,const __nv_bfloat16*,const int*,float*,uint64_t*);' for nb in (1,2,3,4))
for pas,roles in ((0,(0,1)),(1,(0,1,2,3))):
 for role in roles:inst+=f'\ntemplate __global__ void p_epi<{pas},{role},false>(const float*,const __nv_bfloat16*,__nv_bfloat16*,float*,float*,uint64_t*,int);'
 inst+=f'\ntemplate __global__ void p_epi<{pas},0,true>(const float*,const __nv_bfloat16*,__nv_bfloat16*,float*,float*,uint64_t*,int);'
pair_header=epi_header.replace('void p_epi(', 'void p_pair(').replace('float nv0=0,nv1=0;', 'float nv0=input[2048+lane],nv1=0;')
pair_header=pair_header.replace('if constexpr(VECTOR){\n'+VEC, 'if constexpr(VECTOR){\n'+PAIR)
pair_header=pair_header.replace(' uint64_t t1=mark();stamps[lane*2]=t0;', ' if constexpr(PASS==1)asm volatile("st.global.cs.f32 [%0],%1;"::"l"(ss+64+lane),"f"(nv0):"memory");\n uint64_t t1=mark();stamps[lane*2]=t0;')
# The synthetic store above makes the persistent norm accumulator observable inside the marker.
# Discount one STG in the per-unit model; it is absent from the real register-persistent epilogue.
pair_extra=r'''
__global__ void p_store_pair(const float4* p,float* output,uint64_t* stamps){__shared__ float r[1024];int l=threadIdx.x;float4 a=p[l];uint64_t t0=mark();
 *reinterpret_cast<float2*>(r+l*2)=make_float2(a.x,a.z);*reinterpret_cast<float2*>(r+64+l*2)=make_float2(a.y,a.w);__syncthreads();uint64_t t1=mark();
 *reinterpret_cast<float2*>(output+l*2)=*reinterpret_cast<float2*>(r+l*2);*reinterpret_cast<float2*>(output+64+l*2)=*reinterpret_cast<float2*>(r+64+l*2);stamps[l*2]=t0;stamps[l*2+1]=t1;}
template<int ROLE>__global__ void p_norm_flush(const float* input,float* ss,uint64_t* stamps,int block,int ne,int off){int lane=threadIdx.x;float s=input[lane];uint64_t t0=mark();
#pragma unroll
 for(int d=4;d<=16;d*=2)s+=__shfl_xor_sync(~0u,s,d);int n=block*8+(lane&3)*2+ROLE;if(lane<4&&n<ne)atomicAdd(ss+off+n,s);
 uint64_t t1=mark();stamps[lane*2]=t0;stamps[lane*2+1]=t1;}
'''
pair_inst=''
for pas in (0,1):
 for role in (0,1):pair_inst+=f'\ntemplate __global__ void p_pair<{pas},{role},true>(const float*,const __nv_bfloat16*,__nv_bfloat16*,float*,float*,uint64_t*,int);'
for role in (0,1):pair_inst+=f'\ntemplate __global__ void p_norm_flush<{role}>(const float*,float*,uint64_t*,int,int,int);'
# New cuts use the same marked epilogue ABI; expose each persistent sum with the same
# synthetic STG, discounted by the model. Each FOUR role has16 active lanes.
from build_epi_next import source as next_source
next_probes=''
for kind,code,nroles in (('tree',0,2),('four',1,4)):
 full=next_source(kind)[1];a=full.index('    if constexpr(VEC){\n');b=full.index('    }else{\n',a)
 body=full[a+len('    if constexpr(VEC){\n'):b]
 hdr=pair_header.replace('void p_pair(',f'void p_next{code}(').replace(PAIR,body)
 next_probes+=hdr
 for pas in (0,1):
  for role in range(nroles):next_probes+=f'\ntemplate __global__ void p_next{code}<{pas},{role},true>(const float*,const __nv_bfloat16*,__nv_bfloat16*,float*,float*,uint64_t*,int);'
source=head+math+epi_header+extra+inst+pair_header+pair_extra+pair_inst+next_probes+'\n} // namespace\n'
path=DEV/'issue_probe.cu';path.write_text(source)
obj='/tmp/nt1_issue_probe.o';r=subprocess.run(['nvcc','-O3','-std=c++20','-lineinfo','-arch=sm_90a','-c',str(path),'-o',obj,'-Xptxas=-v'],capture_output=True,text=True)
(DEV/'issue_compile.log').write_text(r.stdout+r.stderr);r.check_returncode()
sass=subprocess.run(['cuobjdump','--dump-sass',obj],capture_output=True,text=True,check=True).stdout
(DEV/'issue_probe.sass').write_text(sass)
rows=[]
for name,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
 ins=[l.split(';')[0].split('*/',1)[-1].strip() for l in body.splitlines() if re.match(r'\s*/\*[0-9a-f]{4,}\*/',l)]
 marks=[i for i,l in enumerate(ins) if 'CS2R' in l and 'SR_CLOCK' in l]
 for seg in range(len(marks)//2):
  seq=ins[marks[seg*2]+1:marks[seg*2+1]]
  ops=collections.Counter((l.split()[1] if l.startswith('@') else l.split()[0]).split('.')[0] for l in seq)
  memops=collections.Counter((l.split()[1] if l.startswith('@') else l.split()[0]) for l in seq if any(op in l for op in ('LDS','STS','LDG','REDG','F2FP')))
  rows.append({'name':name,'segment':seg,'instructions':len(seq),'ops':dict(ops),'memory_width_ops':dict(memops),'branches':sum(ops[x] for x in ('BRA','BSSY','BSYNC'))})
for nb in (1,2,3,4):
 row=next(r for r in rows if f'p_mathILi{nb}E' in r['name'] and r['segment']==1)
 assert row['ops'].get('HMMA')==32*nb and row['ops'].get('HMUL2')==128 and row['ops'].get('LDG',0)==16*(nb-1),row
res={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'S14_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'gpu_model':{'sms':132,'clock_GHz_assumed':1.98,'warp_issue_per_sm_cycle_peak':4,'weight_bytes_per16':9,'target_TBs':4.,'warp_instructions_per16_per_lane_budget':4*132*1.98e9/(4e12/(9*32))},'probes':rows}
(DEV/'issue_counts.json').write_text(json.dumps(res,indent=2)+'\n')
for row in rows:print(row)
print('BUDGET',res['gpu_model'])
