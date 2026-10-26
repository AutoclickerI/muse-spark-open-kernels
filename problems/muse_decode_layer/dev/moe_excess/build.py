"""Matched cuts of all three frozen S19 expert branches, with literal BASE retained."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def once(s,a,b,n=1):
 assert s.count(a)==n,(a[:100],s.count(a),n)
 return s.replace(a,b)
def balanced(s,start):
 a=s.index('{',start);depth=1;i=a+1
 while depth:
  if s[i]=='{':depth+=1
  elif s[i]=='}':depth-=1
  i+=1
 return s[start:i],i
PRELUDE=r'''
namespace xlad {
__device__ __forceinline__ uint64_t now(){uint64_t x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;}
__device__ __forceinline__ uint32_t sm(){uint32_t x;asm("mov.u32 %0, %%smid;":"=r"(x));return x;}
__device__ __forceinline__ void sink(uint32_t p,uint32_t a,uint32_t b,uint32_t c,uint32_t d){asm volatile("st.volatile.shared.v2.u32 [%0],{%1,%2};st.volatile.shared.v2.u32 [%0],{%3,%4};"::"r"(p),"r"(a),"r"(b),"r"(c),"r"(d):"memory");}
struct Meter{
 uint64_t *p,entry;uint32_t mask=0;int tiles=0,tokens=0,n32=0,n64=0;
 __device__ Meter(uint64_t*p_,bool on):p(p_){entry=(on&&threadIdx.x==0)?now():0;}
 __device__ void add(int e,int n,int nt){mask|=1u<<e;tiles++;tokens+=n;n32+=nt<=32;n64+=nt>32;}
 __device__ void end(){p[0]=entry;p[1]=now();p[2]=sm();p[3]=tiles;p[4]=mask;p[5]=tokens;p[6]=n32;p[7]=n64;}
};
}
'''

def heavy(cu):
 a=cu.index('template<int NT,bool F,int CS,int S>');tile,end=balanced(cu,a)
 tile=tile.replace('template<int NT,bool F,int CS,int S>','template<int NT,bool F,int CS,int S,int MODE>').replace(' void tile(',' void xtile(')
 tile=once(tile,'unsigned scale_base,float*red){','unsigned scale_base,float*red){\n constexpr bool COPY=MODE!=5&&MODE!=6,EPI=MODE>=3&&MODE!=6,SINK=MODE<=3;')
 # Two KB scratch follows the untouched pipeline allocation. Shared reservations
 # are matched to BASE occupancy for every cut in the host launcher.
 tile=once(tile,'auto*bs=(__nv_bfloat16*)(ocsm+BS);','auto*bs=(__nv_bfloat16*)(ocsm+BS);unsigned sk=smaddr(ocsm+dyn<S,64>()+threadIdx.x*8);\n if constexpr(!COPY){for(int i=threadIdx.x;i<dyn<S,64>();i+=256)ocsm[i]=i<S*WG*QB?(unsigned char)(i+blockIdx.x):(i<BS?0:0x3c);__syncthreads();if constexpr(MODE==6)return;}')
 # Keep exact issue order in L0..L3. MATH retains the same on-chip tile accesses.
 tile=once(tile,'auto issue=[&](int k,int st){','auto issue=[&](int k,int st){if constexpr(COPY){')
 tile=once(tile,'asm volatile("cp.async.commit_group;":::"memory");\n };','asm volatile("cp.async.commit_group;":::"memory");}\n };')
 old='dq(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);keep(af[sub]);'
 new='if constexpr(MODE!=0){'+old+'}else{for(int i=0;i<16;i++)af[sub][i]=((const unsigned*)&q)[i&3];}\n   if constexpr(SINK){xlad::sink(sk,sv.x,sv.y,sv.z,sv.w);for(int i=0;i<4;i++)xlad::sink(sk,af[sub][i*4],af[sub][i*4+1],af[sub][i*4+2],af[sub][i*4+3]);}'
 tile=once(tile,old,new)
 tile=once(tile,'for(int b=0;b<4;++b){if constexpr(NT==16)', 'for(int b=0;b<4;++b)if constexpr(MODE>=2){if constexpr(NT==16)')
 tile=once(tile,'keep(af[0]);keep(af[1]);\n if constexpr(F)', 'keep(af[0]);keep(af[1]);\n if constexpr(SINK){for(int j=0;j<NT/2;j+=4)xlad::sink(sk,__float_as_uint(acc[j]),__float_as_uint(acc[j+1]),__float_as_uint(acc[j+2]),__float_as_uint(acc[j+3]));}\n if constexpr(EPI){\n if constexpr(F)')
 tile=once(tile,' __syncthreads();\n}', ' }\n __syncthreads();\n}')
 tile=tile.replace('asm volatile("wgmma.', 'if constexpr(MODE>=2)asm volatile("wgmma.')
 a=cu.index('template<int DEPTH,bool F>__global__');k,end=balanced(cu,a)
 k=k.replace('template<int DEPTH,bool F>','template<int DEPTH,bool F,int MODE,bool HIST>').replace(' kernel(',' xkernel(')
 k=once(k,'float*Y,float*SS,int M){','float*Y,float*SS,int M,uint64_t*trace){')
 k=once(k,'float*red=(float*)ocsm+(threadIdx.x>>7)*256;', 'float*red=(float*)ocsm+(threadIdx.x>>7)*256;\n xlad::Meter meter(trace+blockIdx.x*8,HIST);')
 k=k.replace('tile<NT,F,1,S>','xtile<NT,F,1,S,MODE>')
 k=once(k,'for(int row=0;row<n;row+=64){int rem=min(n-row,64);','for(int row=0;row<n;row+=64){int rem=min(n-row,64);if constexpr(HIST)if(threadIdx.x==0)meter.add(e,rem,rem<=32?32:64);')
 k=k[:-1]+' if constexpr(HIST)if(threadIdx.x==0)meter.end();\n}'
 tile=tile.replace('GS[e]*256.f','(MODE==5?1.f:GS[e]*256.f)').replace('GS[e+32]*256.f','(MODE==5?1.f:GS[e+32]*256.f)').replace('__bfloat162float(PNW[cc])','(MODE==5?0.f:__bfloat162float(PNW[cc]))')
 return '\nnamespace chunk {\n'+tile+'\n'+k+'\n}\n'

def medium(cu):
 a=cu.index('template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>');k,end=balanced(cu,a)
 k=k.replace('template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>','template<int NT,bool FIRST,int VEC,int FM,int LO,int HI,int MODE,bool HIST>').replace(' gemm(',' xgemm(')
 k=once(k,'int M,unsigned scale_base){','int M,unsigned scale_base,uint64_t*trace){\n constexpr bool COPY=MODE!=5&&MODE!=6,EPI=MODE>=3&&MODE!=6,SINK=MODE<=3;')
 k=once(k,'int t=threadIdx.x,w=t>>5,l=t&31;', 'int t=threadIdx.x,w=t>>5,l=t&31;unsigned sk=smaddr(typed_sm+((smem_bytes<16>()+15)&~15)+t*8);xlad::Meter meter(trace+blockIdx.x*8,HIST);\n if constexpr(!COPY){for(int i=t;i<smem_bytes<16>();i+=128)typed_sm[i]=i<S*QB?(unsigned char)(i+blockIdx.x):(i<BS?0:0x3c);__syncthreads();}')
 k=once(k,'int src[NJ];','if constexpr(HIST)if(t==0)meter.add(e,min(NT,count-row),NT);\n  if constexpr(MODE==6)continue;\n  int src[NJ];')
 k=once(k,'auto issue=[&](int slot){','auto issue=[&](int slot){if constexpr(COPY){')
 k=once(k,'for(int j=0;j<NJ;++j)xg[j]+=KT;\n   };','for(int j=0;j<NJ;++j)xg[j]+=KT;}\n   };')
 # Only VEC7/6 are instantiated, so the non-WIDE / old loop remains unchanged.
 old='dq(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);'
 k=once(k,old,'if constexpr(MODE!=0){'+old+'}else{for(int i=0;i<16;i++)af[sub][i]=((const unsigned*)&q)[i&3];}\n       if constexpr(SINK){xlad::sink(sk,sv.x,sv.y,sv.z,sv.w);for(int i=0;i<4;i++)xlad::sink(sk,af[sub][i*4],af[sub][i*4+1],af[sub][i*4+2],af[sub][i*4+3]);}')
 k=k.replace('asm volatile("wgmma.', 'if constexpr(MODE>=2)asm volatile("wgmma.')
 k=k.replace('mma_group<NT>(acc,','if constexpr(MODE>=2)mma_group<NT>(acc,')
 k=once(k,'keep(af[0]);keep(af[1]);\n  if constexpr(FIRST', 'keep(af[0]);keep(af[1]);\n  if constexpr(SINK){for(int j=0;j<NT/2;j+=4)xlad::sink(sk,__float_as_uint(acc[j]),__float_as_uint(acc[j+1]),__float_as_uint(acc[j+2]),__float_as_uint(acc[j+3]));}\n  if constexpr(EPI){\n  if constexpr(FIRST')
 k=once(k,'  __syncthreads();\n }\n}', '  }\n  __syncthreads();\n }\n if constexpr(HIST)if(t==0)meter.end();\n}')
 k=k.replace('GS[e]*256.f','(MODE==5?1.f:GS[e]*256.f)').replace('GS[e+32]*256.f','(MODE==5?1.f:GS[e+32]*256.f)').replace('__bfloat162float(PNW[c])','(MODE==5?0.f:__bfloat162float(PNW[c]))')
 return '\nnamespace mwbr {\n'+k+'\n}\n'

def low(cu):
 a=cu.index('template<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>');k,end=balanced(cu,a)
 k=k.replace('template<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>','template<int NT,bool FIRST,bool SMALL,bool ZSS,int THR,int MODE,bool HIST>').replace(' phase(',' xphase(')
 k=once(k,'int M,int nstage){','int M,int nstage,uint64_t*trace){\n constexpr bool COPY=MODE!=5&&MODE!=6,EPI=MODE>=3&&MODE!=6,SINK=MODE<=3;')
 k=once(k,'const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;', 'const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;unsigned sk=smem_u32(smem+nstage*TILE_BYTES+2*NCW*32*RED*sizeof(float)+2*nstage*8+sizeof(Route)+((M*8+15)&~15)+4*M*8+tid*8);xlad::Meter meter(trace+c*8,HIST);')
 # A forward declaration is provided before the clone for smem_bytes.
 k=k.replace('prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);','if constexpr(COPY)prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);')
 k=once(k,'  const int q = lane & 3;', '  if constexpr(!COPY){for(int i=tid;i<nstage*TILE_BYTES;i+=NTHR)stages[i]=i%TILE_BYTES<TILE_W_BYTES?(unsigned char)(i+c):0;__syncthreads();}\n  const int q = lane & 3;')
 k=once(k,'      const int ne = rt->n[e], off = rt->off[e];', '      const int ne = rt->n[e], off = rt->off[e];if constexpr(HIST)if(tid==0)meter.add(e,min(NB,ne-nb*NB),NB);\n      if constexpr(MODE==6){advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);continue;}')
 k=once(k,'if (pass != cur_pass || e != cur_e || nb != cur_nb)', 'if (pass != cur_pass || e != cur_e || nb != cur_nb)')
 k=once(k,'            if (row) {','            if constexpr(COPY)if (row) {')
 k=once(k,'uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;', 'uint4 v0 = make_uint4(0x3c003c00,0x3c003c00,0x3c003c00,0x3c003c00), v1 = v0;')
 k=once(k,'            Bf[j][g][0] = v0.x;', '            if constexpr(SINK){xlad::sink(sk,v0.x,v0.y,v0.z,v0.w);xlad::sink(sk,v1.x,v1.y,v1.z,v1.w);}\n            Bf[j][g][0] = v0.x;')
 k=once(k,'      mbar_wait(&fullb[st], ph);','      if constexpr(COPY)mbar_wait(&fullb[st], ph);')
 k=once(k,'      if (lane == 0) mbar_arrive(&emptyb[st]);','      if constexpr(COPY)if (lane == 0) mbar_arrive(&emptyb[st]);')
 k=once(k,'          const uint32_t sw = scw[g >> 1];','          if constexpr(SINK)xlad::sink(sk,scv.x,scv.y,scv.z,scv.w);\n          const uint32_t sw = scw[g >> 1];')
 k=once(k,'            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);', '            const uint32_t a0 = MODE==0?wv.x:hmul2(os[0], Sr), a1 = MODE==0?wv.y:hmul2(os[1], Sr8);\n            const uint32_t a2 = MODE==0?wv.z:hmul2(os[2], Sr), a3 = MODE==0?wv.w:hmul2(os[3], Sr8);\n            if constexpr(SINK)xlad::sink(sk,a0,a1,a2,a3);')
 k=once(k,'for (int j = 0; j < NT; ++j) mma16816(', 'if constexpr(MODE>=2)for (int j = 0; j < NT; ++j) mma16816(')
 k=once(k,'      const float* rb = red +', '      if constexpr(SINK){for(int j=0;j<NT;j++)xlad::sink(sk,__float_as_uint(acc[j][0]),__float_as_uint(acc[j][1]),__float_as_uint(acc[j][2]),__float_as_uint(acc[j][3]));}\n      if constexpr(EPI){\n      const float* rb = red +')
 k=once(k,'      buf ^= 1;', '      }\n      buf ^= 1;')
 k=k[:-1]+' if constexpr(HIST)if(tid==0)meter.end();\n}'
 k=k.replace('gs13[e] * 256.f','(MODE==5?1.f:gs13[e]*256.f)').replace('gs13[e+32]*256.f','(MODE==5?1.f:gs13[e+32]*256.f)').replace('gs2[e] * 256.f','(MODE==5?1.f:gs2[e]*256.f)').replace('1.f + __bfloat162float(pnw[m * 16 + (tid & 15)])','(MODE==5?1.f:1.f+__bfloat162float(pnw[m*16+(tid&15)]))')
 return '\nnamespace pmz {\n'+k+'\n}\n'

def main():
 (H/'out').mkdir(parents=True,exist_ok=True)
 frozen=json.loads((H.parent/'s19/frozen.json').read_text());packed=(R/frozen['file']).read_text();t=ast.parse(packed);a=next(n for n in t.body if isinstance(n,ast.Assign) and any(getattr(x,'id','')=='_source' for x in n.targets));blob=next(n.value for n in ast.walk(a.value) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000);source=lzma.decompress(base64.b85decode(blob)).decode();assert sha(source)==frozen['expanded_sha256']
 vals={}
 for n in ast.parse(source).body:
  if isinstance(n,ast.Assign) and any(getattr(x,'id','') in ('P3B_CUDA','P3B_CPP','MT_PMZ_CUDA') for x in n.targets):vals[n.targets[0].id]=ast.literal_eval(n.value)
 base=vals['P3B_CUDA'];pm=vals['MT_PMZ_CUDA'];cu=base+PRELUDE+heavy(base)+medium(base)+low(base)+(H/'host.cu.inc').read_text();pmcu=pm+PRELUDE+low(pm)+(H/'host_pm.cu.inc').read_text()
 (H/'out/ladder.cu').write_text(cu);(H/'out/ladder_pm.cu').write_text(pmcu)
 checker='\n'.join(l for p in ('checks.py','nvfp4.py','reference.py') for l in (R/'problems/muse_moe_w4a16'/p).read_text().splitlines() if not l.startswith(('from checks import','from nvfp4 import','from __future__ import'))).replace('leading=(N_EXPERTS,)','leading=(32,)')
 helpers=(H.parent/'moe_public_ports/runtime.py.inc').read_text();helpers=helpers[:helpers.index('def lee_module')]+helpers[helpers.index('def s19_prepare'):]
 layer='\n'.join(l for p in ('checks.py','nvfp4.py','reference.py') for l in (R/'problems/muse_decode_layer'/p).read_text().splitlines() if not l.startswith(('from checks import','from nvfp4 import','from __future__ import')))
 B=dict(layer=layer,source=source,checker=checker,cuda=cu,pmcuda=pmcu,cpp=vals['P3B_CPP'],helpers=helpers)
 text='import base64,gc,hashlib,json,linecache,lzma,math,statistics,struct,time,types\nimport torch\nB=json.loads(lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(json.dumps(B).encode(),preset=9)).decode())+')))\n'+helpers+'\n'+(H/'runtime.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text();compile(text,'<large-m-ladder>','exec');(H/'out/ladder.py').write_text(text)
 (H/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(source),cuda_sha256=sha(cu),pmcuda_sha256=sha(pmcu),identities_sha256=sha(source+cu+pmcu),payload_sha256=sha(text),weight_seeds=[381522721+7919*i for i in range(8)],seeds=[381522727+104729*i for i in range(8)],rounds=4),indent=2)+'\n');print('ladder',len(text),'CUDA',sha(cu)[:12],sha(pmcu)[:12])
if __name__=='__main__':main()
