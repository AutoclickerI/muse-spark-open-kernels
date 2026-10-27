"""Matched load-path split of S22's large-M MoE (identical to S19's) into reuse, activation,
issue, residency and in-flight parts. Two lab payloads share one generated CUDA image:
J (all three branches together, the production structure) and I (each branch alone).
"""
import ast, base64, hashlib, json, lzma, subprocess, sys
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[4]
BASE_SHA = '4b8c803db080f4e0f2a69e4c666ee91573fc6ffde238d2e63db4223f83f84d27'
BASE_PACKED = 'd1ccdc358ef5fb0f038a1d0e741465f512f498e90d39a16e56f3f0b0e47615f3'
BASE_REVISION = 'daa7a1c3b902e80537e2e50f1c7d6c6ea52fb527'
WEIGHT_SEEDS = [381522721 + 7919 * i for i in range(8)]
SEEDS = [381522727 + 104729 * i for i in range(8)]


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def once(s, a, b, n=1):
    assert s.count(a) == n, (a[:100], s.count(a), n)
    return s.replace(a, b)


def balanced(s, start):
    a = s.index('{', start); depth = 1; i = a + 1
    while depth:
        if s[i] == '{': depth += 1
        elif s[i] == '}': depth -= 1
        i += 1
    return s[start:i], i


def s22_source():
    packed = subprocess.check_output(['git', 'show', BASE_REVISION + ':problems/muse_decode_layer/cand_r3_best7_s22_final.py'], cwd=R)
    assert sha(packed) == BASE_PACKED
    call = next(n for n in ast.walk(ast.parse(packed)) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'b85decode')
    source = lzma.decompress(base64.b85decode(ast.literal_eval(call.args[0]))).decode()
    assert sha(source) == BASE_SHA
    return source


# Modes. -2 exact original kernel (BASE/AA). 0 ladder L0 (sinked raw operands, receipt 1b75cd73).
# 4 full clone (HIST observer). 7 unique-byte stream. Loads-only, no sinks, original ring sync:
# 8 L0N, 9 NOX (no activation copies), 10 REUSE (heavy weights once per CTA tile, not per 64-row
# chunk), 11 ISSUE (heavy weights as two per-warpgroup bulk copies per stage), 13 PFN (L0N plus
# in-kernel L2 prefetch of the weight stream ahead of the ring). 12 PF: full math plus that prefetch.
FLAGS = ('constexpr bool COPY=true,SINK=MODE==0,LN=MODE>=8&&MODE!=12,XL=MODE!=9,WONCE=MODE==10,'
         'BULK=MODE==11,PF=MODE==12||MODE==13,MMA=MODE==4||MODE==12,EPI=MODE==4||MODE==12;')
PRELUDE = r'''
namespace xlad {
__device__ __forceinline__ uint64_t now(){uint64_t x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;}
__device__ __forceinline__ uint32_t sm(){uint32_t x;asm("mov.u32 %0, %%smid;":"=r"(x));return x;}
__device__ __forceinline__ void sink(uint32_t p,uint32_t a,uint32_t b,uint32_t c,uint32_t d){asm volatile("st.volatile.shared.v2.u32 [%0],{%1,%2};st.volatile.shared.v2.u32 [%0],{%3,%4};"::"r"(p),"r"(a),"r"(b),"r"(c),"r"(d):"memory");}
__device__ __forceinline__ void l2pf(const void*p,unsigned bytes){asm volatile("cp.async.bulk.prefetch.L2.global [%0],%1;"::"l"(p),"r"(bytes):"memory");}
__device__ __forceinline__ void bulk(unsigned dst,const void*src,unsigned bytes,unsigned bar){asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(dst),"l"(src),"r"(bytes),"r"(bar):"memory");}
struct Meter{
 uint64_t *p,entry;uint32_t mask=0;int tiles=0,tokens=0,n32=0,n64=0;
 __device__ Meter(uint64_t*p_,bool on):p(p_){entry=(on&&threadIdx.x==0)?now():0;}
 __device__ void add(int e,int n,int nt){mask|=1u<<e;tiles++;tokens+=n;n32+=nt<=32;n64+=nt>32;}
 __device__ void end(){p[0]=entry;p[1]=now();p[2]=sm();p[3]=tiles;p[4]=mask;p[5]=tokens;p[6]=n32;p[7]=n64;}
};
}
'''


def heavy(cu):
    a = cu.index('template<int NT,bool F,int CS,int S>'); tile, _ = balanced(cu, a)
    tile = tile.replace('template<int NT,bool F,int CS,int S>', 'template<int NT,bool F,int CS,int S,int MODE>').replace(' void tile(', ' void xtile(')
    tile = once(tile, 'unsigned scale_base,float*red){', 'unsigned scale_base,float*red,unsigned&phase){\n ' + FLAGS)
    # Scratch beyond the ring: per-thread sink words (MODE 0) or the BULK stage barriers.
    tile = once(tile, 'auto*bs=(__nv_bfloat16*)(ocsm+BS);', 'auto*bs=(__nv_bfloat16*)(ocsm+BS);unsigned sk=smaddr(ocsm+dyn<S,64>()+threadIdx.x*8),mbar=smaddr(ocsm+dyn<S,64>());')
    tile = once(tile, 'auto issue=[&](int k,int st){\n  if(wg==0)', 'auto issue=[&](int k,int st){\n  if constexpr(XL)if(wg==0)')
    old_w = ('  if(t<64){\n   #pragma unroll\n   for(int g=0;g<4;++g)asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(qd+st*WG*QB+(g>>1)*2048+(g&1)*256),"l"(qg+g*64):"memory");\n   qg+=256;\n  }\n'
             '  if(t>=64&&t<96){asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(sd+st*WG*SB),"l"(sg):"memory");sg+=128;}\n')
    # PF: one thread per 16-row group prefetches its contiguous 1 KiB/stage Q and 128 B/stage scale
    # rows 8-12 stages ahead of the ring (stages [0,12) at the tile's first issue).
    pf = ('  if constexpr(PF)if((k&3)==0){int lo=k?k+8:0,hi=min(32,k+12);if(hi>lo){\n'
          '   if(t<64&&(t&15)==0)xlad::l2pf(qg+(lo-k)*256,(hi-lo)*1024);\n'
          '   if(t>=64&&t<96&&((t-64)&7)==0)xlad::l2pf(sg+(lo-k)*128,(hi-lo)*128);}}\n')
    # BULK: the per-warpgroup 128 KiB Q and 16 KiB scale regions of this tile are contiguous; the
    # modelled relayout reads them as one 4 KiB + one 512 B bulk copy per stage (same bytes).
    bulk = ('  if constexpr(BULK){if(t==0){unsigned bar=mbar+st*8;\n'
            '   asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar),"r"(QB+SB):"memory");\n'
            '   xlad::bulk(smaddr(qs+(st*WG+wg)*QB),(const unsigned char*)wq0+k*QB,QB,bar);xlad::bulk(smaddr(ss+(st*WG+wg)*SB),ws0+k*SB,SB,bar);}}\n'
            '  else if(!WONCE||row==0){\n')
    tile = once(tile, old_w, pf + bulk + old_w.replace('\n  if(t>=64', '\n  if(t>=64') + '  }\n')
    tile = once(tile, ' const int*qg=Q;const unsigned char*sg=SC;', ' const int*qg=Q;const unsigned char*sg=SC;const int*wq0=Q+((unsigned)e*(N/16)+mt*4)*128*64;const unsigned char*ws0=SC+((unsigned)e*(N/16)+mt*4)*128*32;')
    fence = '#if BR_HOPPER\n  asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n'
    tile = once(tile, fence, fence + '  if constexpr(LN){if constexpr(BULK){asm volatile("{ .reg .pred q; XW%=: mbarrier.try_wait.parity.shared::cta.b64 q,[%0],%1; @!q bra XW%=; }"::"r"(mbar+st*8),"r"((phase>>st)&1u):"memory");phase^=1u<<st;}\n'
                     '   __syncthreads();if(k+S-1<32)issue(k+S-1,(st+S-1)%S);else asm volatile("cp.async.commit_group;":::"memory");continue;}\n')
    old = 'dq(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);keep(af[sub]);'
    tile = once(tile, old, 'if constexpr(MODE!=0){' + old + '}else{for(int i=0;i<16;i++)af[sub][i]=((const unsigned*)&q)[i&3];}\n   if constexpr(SINK){xlad::sink(sk,sv.x,sv.y,sv.z,sv.w);for(int i=0;i<4;i++)xlad::sink(sk,af[sub][i*4],af[sub][i*4+1],af[sub][i*4+2],af[sub][i*4+3]);}')
    tile = once(tile, 'for(int b=0;b<4;++b){if constexpr(NT==16)', 'for(int b=0;b<4;++b)if constexpr(MMA){if constexpr(NT==16)')
    tile = once(tile, 'keep(af[0]);keep(af[1]);\n if constexpr(F)', 'keep(af[0]);keep(af[1]);\n if constexpr(SINK){for(int j=0;j<NT/2;j+=4)xlad::sink(sk,__float_as_uint(acc[j]),__float_as_uint(acc[j+1]),__float_as_uint(acc[j+2]),__float_as_uint(acc[j+3]));}\n if constexpr(EPI){\n if constexpr(F)')
    tile = once(tile, ' __syncthreads();\n}', ' }\n __syncthreads();\n}')
    tile = tile.replace('asm volatile("wgmma.', 'if constexpr(MMA)asm volatile("wgmma.')
    a = cu.index('template<int DEPTH,bool F>__global__'); k, _ = balanced(cu, a)
    k = k.replace('template<int DEPTH,bool F>', 'template<int DEPTH,bool F,int MODE,bool HIST>').replace(' kernel(', ' xkernel(')
    k = once(k, 'float*Y,float*SS,int M){', 'float*Y,float*SS,int M,uint64_t*trace){')
    k = once(k, 'float*red=(float*)ocsm+(threadIdx.x>>7)*256;',
             'float*red=(float*)ocsm+(threadIdx.x>>7)*256;unsigned phase=0;\n xlad::Meter meter(trace+blockIdx.x*8,HIST);\n'
             ' if constexpr(MODE==11){if(threadIdx.x==0){for(int s=0;s<S;++s)asm volatile("mbarrier.init.shared::cta.b64 [%0],2;"::"r"(smaddr(ocsm+dyn<S,64>()+s*8)):"memory");\n'
             '  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}__syncthreads();}')
    k = k.replace('tile<NT,F,1,S>', 'xtile<NT,F,1,S,MODE>').replace('0x70007000u,red)', '0x70007000u,red,phase)')
    k = once(k, 'for(int row=0;row<n;row+=64){int rem=min(n-row,64);', 'for(int row=0;row<n;row+=64){int rem=min(n-row,64);if constexpr(HIST)if(threadIdx.x==0)meter.add(e,rem,rem<=32?32:64);')
    k = k[:-1] + ' if constexpr(HIST)if(threadIdx.x==0)meter.end();\n}'
    return '\nnamespace chunk {\n' + tile + '\n' + k + '\n}\n'


def medium(cu):
    a = cu.index('template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>'); k, _ = balanced(cu, a)
    k = k.replace('template<int NT,bool FIRST,int VEC,int FM,int LO,int HI>', 'template<int NT,bool FIRST,int VEC,int FM,int LO,int HI,int MODE,bool HIST>').replace(' gemm(', ' xgemm(')
    k = once(k, 'int M,unsigned scale_base){', 'int M,unsigned scale_base,uint64_t*trace){\n ' + FLAGS)
    k = once(k, 'int t=threadIdx.x,w=t>>5,l=t&31;', 'int t=threadIdx.x,w=t>>5,l=t&31;unsigned sk=smaddr(typed_sm+((smem_bytes<16>()+15)&~15)+t*8);xlad::Meter meter(trace+blockIdx.x*8,HIST);')
    k = once(k, 'int src[NJ];', 'if constexpr(HIST)if(t==0)meter.add(e,min(NT,count-row),NT);\n  int src[NJ];')
    k = once(k, 'auto issue=[&](int slot){', 'int pk=0;auto issue=[&](int slot){'
             'if constexpr(PF)if((pk&3)==0){int lo=pk?pk+8:0,hi=min(NS,pk+12);if(hi>lo){if(wq&&(t&15)==0)xlad::l2pf(qg+(lo-pk)*256,(hi-lo)*1024);if(ws&&((t-64)&7)==0)xlad::l2pf(sg+(lo-pk)*128,(hi-lo)*128);}}++pk;')
    k = once(k, '    #pragma unroll\n    for(int c=0;c<KT/64;++c){\n     #pragma unroll\n     for(int j=0;j<NJ;++j)asm volatile("cp.async.cg',
             '    if constexpr(XL){\n    #pragma unroll\n    for(int c=0;c<KT/64;++c){\n     #pragma unroll\n     for(int j=0;j<NJ;++j)asm volatile("cp.async.cg')
    k = once(k, '    }\n    #pragma unroll\n    for(int j=0;j<NJ;++j)xg[j]+=KT;', '    }}\n    #pragma unroll\n    for(int j=0;j<NJ;++j)xg[j]+=KT;')
    fence = '#if BR_HOPPER\n     if constexpr(VEC<4||WIDE)asm volatile("fence.proxy.async.shared::cta;":::"memory");  // VEC 4/5: timing probe without it\n#endif\n'
    k = once(k, fence, fence + '     if constexpr(LN){if(k0+st+S-1<NS)issue((st+S-1)%S);asm volatile("cp.async.commit_group;":::"memory");continue;}\n')
    old = 'dq(q.x,q.y,sub?sv.z:sv.x,af[sub],scale_base);dq(q.z,q.w,sub?sv.w:sv.y,af[sub]+8,scale_base);'
    k = once(k, old, 'if constexpr(MODE!=0){' + old + '}else{for(int i=0;i<16;i++)af[sub][i]=((const unsigned*)&q)[i&3];}\n       if constexpr(SINK){xlad::sink(sk,sv.x,sv.y,sv.z,sv.w);for(int i=0;i<4;i++)xlad::sink(sk,af[sub][i*4],af[sub][i*4+1],af[sub][i*4+2],af[sub][i*4+3]);}')
    k = k.replace('asm volatile("wgmma.', 'if constexpr(MMA)asm volatile("wgmma.')
    k = k.replace('mma_group<NT>(acc,', 'if constexpr(MMA)mma_group<NT>(acc,')
    k = once(k, 'keep(af[0]);keep(af[1]);\n  if constexpr(FIRST', 'keep(af[0]);keep(af[1]);\n  if constexpr(SINK){for(int j=0;j<NT/2;j+=4)xlad::sink(sk,__float_as_uint(acc[j]),__float_as_uint(acc[j+1]),__float_as_uint(acc[j+2]),__float_as_uint(acc[j+3]));}\n  if constexpr(EPI){\n  if constexpr(FIRST')
    k = once(k, '  __syncthreads();\n }\n}', '  }\n  __syncthreads();\n }\n if constexpr(HIST)if(t==0)meter.end();\n}')
    return '\nnamespace mwbr {\n' + k + '\n}\n'


def low(cu):
    a = cu.index('template<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>'); k, _ = balanced(cu, a)
    k = k.replace('template<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>', 'template<int NT,bool FIRST,bool SMALL,bool ZSS,int THR,int MODE,bool HIST>').replace(' phase(', ' xphase(')
    k = once(k, 'int M,int nstage){', 'int M,int nstage,uint64_t*trace){\n ' + FLAGS)
    k = once(k, 'const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;', 'const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;unsigned sk=smem_u32(smem+nstage*TILE_BYTES+2*NCW*32*RED*sizeof(float)+2*nstage*8+sizeof(Route)+((M*8+15)&~15)+4*M*8+tid*8);xlad::Meter meter(trace+c*8,HIST);')
    k = k.replace('prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);', 'xpf<PF>(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);')
    k = once(k, '      const int ne = rt->n[e], off = rt->off[e];', '      const int ne = rt->n[e], off = rt->off[e];if constexpr(HIST)if(tid==0)meter.add(e,min(NB,ne-nb*NB),NB);')
    k = once(k, '            if (row) {', '            if constexpr(XL)if (row) {')
    k = once(k, '            Bf[j][g][0] = v0.x;', '            if constexpr(SINK){xlad::sink(sk,v0.x,v0.y,v0.z,v0.w);xlad::sink(sk,v1.x,v1.y,v1.z,v1.w);}\n'
             '            if constexpr(LN)asm volatile(""::"r"(v0.x),"r"(v0.y),"r"(v0.z),"r"(v0.w),"r"(v1.x),"r"(v1.y),"r"(v1.z),"r"(v1.w));\n            Bf[j][g][0] = v0.x;')
    k = once(k, '      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile)', '      if constexpr(!LN){\n        const uint4* wq = reinterpret_cast<const uint4*>(tile)')
    k = once(k, '          const uint32_t sw = scw[g >> 1];', '          if constexpr(SINK)xlad::sink(sk,scv.x,scv.y,scv.z,scv.w);\n          const uint32_t sw = scw[g >> 1];')
    k = once(k, '            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);', '            const uint32_t a0 = MODE==0?wv.x:hmul2(os[0], Sr), a1 = MODE==0?wv.y:hmul2(os[1], Sr8);\n            const uint32_t a2 = MODE==0?wv.z:hmul2(os[2], Sr), a3 = MODE==0?wv.w:hmul2(os[3], Sr8);\n            if constexpr(SINK)xlad::sink(sk,a0,a1,a2,a3);')
    k = once(k, 'for (int j = 0; j < NT; ++j) mma16816(', 'if constexpr(MMA)for (int j = 0; j < NT; ++j) mma16816(')
    k = once(k, 'for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j]', 'if constexpr(!LN)for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j]')
    k = once(k, '      const float* rb = red +', '      if constexpr(SINK){for(int j=0;j<NT;j++)xlad::sink(sk,__float_as_uint(acc[j][0]),__float_as_uint(acc[j][1]),__float_as_uint(acc[j][2]),__float_as_uint(acc[j][3]));}\n      if constexpr(EPI){\n      const float* rb = red +')
    k = once(k, '      buf ^= 1;', '      }\n      buf ^= 1;')
    k = k[:-1] + ' if constexpr(HIST)if(tid==0)meter.end();\n}'
    # PF: the single producer also prefetches the tile after the one it just issued into L2.
    a = cu.index('__device__ __forceinline__ void prefetch_one('); p, _ = balanced(cu, a)
    p = p.replace('void prefetch_one(', 'void prefetch_one_pf(')
    p = once(p, '  if (++st == nstage) { st=0; ph^=1; }', '  if (idx < total) xlad::l2pf((pass?w2p:w13p)+(static_cast<size_t>(e)*(pass?NT2:NT1)+m)*TILE_BYTES,TILE_BYTES);\n  if (++st == nstage) { st=0; ph^=1; }')
    disp = ('template<bool P>__device__ __forceinline__ void xpf(Route* rt,int total,int first,int a2,int nstage,uint8_t* stages,uint64_t* fullb,uint64_t* emptyb,const uint8_t* w13p,const uint8_t* w2p){\n'
            ' if constexpr(P)prefetch_one_pf(rt,total,first,a2,nstage,stages,fullb,emptyb,w13p,w2p);else prefetch_one(rt,total,first,a2,nstage,stages,fullb,emptyb,w13p,w2p);}\n')
    return '\nnamespace pmz {\n' + p + '\n' + disp + k + '\n}\n'


def main():
    out = H / 'out'; out.mkdir(parents=True, exist_ok=True)
    source = s22_source()
    vals = {}
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and any(getattr(x, 'id', '') in ('P3B_CUDA', 'P3B_CPP', 'MT_PMZ_CUDA') for x in n.targets):
            vals[n.targets[0].id] = ast.literal_eval(n.value)
    base, pm = vals['P3B_CUDA'], vals['MT_PMZ_CUDA']
    cu = base + PRELUDE + heavy(base) + medium(base) + low(base) + (H / 'host.cu.inc').read_text()
    pmcu = pm + PRELUDE + low(pm) + (H / 'host_pm.cu.inc').read_text()
    (out / 'split.cu').write_text(cu); (out / 'split_pm.cu').write_text(pmcu)
    checker = '\n'.join(l for p in ('checks.py', 'nvfp4.py', 'reference.py') for l in (R / 'problems/muse_moe_w4a16' / p).read_text().splitlines() if not l.startswith(('from checks import', 'from nvfp4 import', 'from __future__ import'))).replace('leading=(N_EXPERTS,)', 'leading=(32,)')
    helpers = (H.parent.parent / 'moe_public_ports/runtime.py.inc').read_text()
    helpers = helpers[:helpers.index('def lee_module')] + helpers[helpers.index('def s19_prepare'):]
    layer = '\n'.join(l for p in ('checks.py', 'nvfp4.py', 'reference.py') for l in (R / 'problems/muse_decode_layer' / p).read_text().splitlines() if not l.startswith(('from checks import', 'from nvfp4 import', 'from __future__ import')))
    B = dict(layer=layer, source=source, checker=checker, cuda=cu, pmcuda=pmcu, cpp=vals['P3B_CPP'], helpers=helpers)
    blob = repr(base64.b85encode(lzma.compress(json.dumps(B).encode(), preset=9)).decode())
    runtime = (H / 'runtime.py.inc').read_text(); lab = (H / 'lab.py.inc').read_text()
    manifest = dict(source_sha256=sha(source), cuda_sha256=sha(cu), pmcuda_sha256=sha(pmcu), identities_sha256=sha(source + cu + pmcu),
                    weight_seeds=WEIGHT_SEEDS, seeds=SEEDS, rounds=4, payloads={})
    for name in ('J', 'I', 'I2'):
        # I2's arm table follows the shared runtime, so the sent J and I payloads stay byte-identical.
        extra = (H / 'arms_I2.py.inc').read_text() + '\n' if name == 'I2' else ''
        text = ('import base64,gc,hashlib,json,linecache,lzma,math,statistics,struct,time,types\nimport torch\n'
                f'ARMSET={name!r}\nB=json.loads(lzma.decompress(base64.b85decode(' + blob + ')))\n' + helpers + '\n' + runtime + '\n' + extra + lab)
        compile(text, f'<split-{name}>', 'exec')
        (out / f'split_{name}.py').write_text(text)
        manifest['payloads'][name] = sha(text)
    (H / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('split', {k: v[:12] for k, v in manifest['payloads'].items()}, 'CUDA', sha(cu)[:12], sha(pmcu)[:12])


if __name__ == '__main__':
    main()
