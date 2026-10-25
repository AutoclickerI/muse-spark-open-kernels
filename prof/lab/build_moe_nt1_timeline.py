"""Freeze matched graph-node/CTA timelines and the shared router metadata experiment."""
import hashlib
import re
from pathlib import Path

from build_moe_nt1_limit import literal
from build_moe_small_debug import select

ROOT=Path(__file__).resolve().parents[2]
DEV=ROOT/'problems/muse_decode_layer/dev/nt1_timeline'


def timeline(source):
    source=source.replace('moe_fused','timed_moe')
    source=source.replace('int M, int local_start, int nstage) {','int M, int local_start, int nstage, uint64_t* trace) {')
    source=source.replace('(__nv_bfloat16*)v[12],M,start,stages);','(__nv_bfloat16*)v[12],M,start,stages,(uint64_t*)v[13]);')
    at=source.index('template <int NT>\n__global__')
    source=source[:at]+'''__device__ __forceinline__ uint64_t tg(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}
'''+source[at:]
    old='  constexpr int NB = 8 * NT;'
    source=source.replace(old,'''  __shared__ uint64_t probe[64];
  const int pt=threadIdx.x;
  if((pt&31)==0)probe[32+(pt>>5)]=tg();
  if(pt==0){probe[0]=probe[32];probe[20]=clock64();}
  if(pt<64 && pt!=0 && pt!=20 && !(pt>=32&&pt<40))probe[pt]=0;
'''+old,1)
    source=source.replace('''    for (int i=0;i<nstage;++i)
      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);''',
    '''    probe[10]=tg();
    for (int i=0;i<nstage;++i)
      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);''')
    routed='  if(tid==0){probe[1]=tg();probe[23]=c<2*M;uint32_t sm;asm("mov.u32 %0, %%smid;":"=r"(sm));probe[24]=sm;probe[25]=G;probe[26]=b1-a1;probe[27]=b2-a2;}\n'
    if '  // zero this CTA' in source:source=source.replace('  // zero this CTA',routed+'  // zero this CTA')
    else:source=source.replace('  const int q = lane & 3;',routed+'  const int q = lane & 3;')
    source=source.replace('  const int q = lane & 3;', '  if(tid==0)probe[2]=tg();\n  const int q = lane & 3;')
    source=source.replace('''          while (ld_acquire(cnt + C_DONE + e) < target)''',
        '''          uint64_t wait0=tid==0?tg():0;
          while (ld_acquire(cnt + C_DONE + e) < target)''')
    source=source.replace('''          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);''',
        '''          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);
          if(tid==0)probe[16]+=tg()-wait0;''')
    source=source.replace('      if (pass == 1 && tid < 16 * NB)',
        '      if(tid==0 && pass==1 && u==ua)probe[4]=tg();\n      if (pass == 1 && tid < 16 * NB)')
    source=source.replace('''      mbar_wait(&fullb[st], ph);''','''      if(tid==0 && u==ua)probe[pass?14:11]=tg();
      mbar_wait(&fullb[st], ph);
      if(tid==0 && u==ua)probe[pass?15:12]=tg();''')
    source=source.replace('      const float* rb = red + buf * NCW * 32 * RED;', '      const float* rb = red + buf * NCW * 32 * RED;\n      uint64_t norm0=(tid==0 && pass==1)?tg():0;')
    source=source.replace('''      buf ^= 1;''','''      if(tid==0 && pass==1)probe[17]+=tg()-norm0;
      if(tid==0 && pass==0 && u==ua)probe[13]=tg();
      buf ^= 1;''')
    old='''      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
    }
  }
  if (cur_pass == 0'''
    new='''      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);
    }
    if(tid==0)probe[pass?5:3]=tg();
  }
  if (cur_pass == 0'''
    assert old in source;source=source.replace(old,new)
    source=source.replace('''  if (tid == 0) {
    if (atomicAdd(cnt + C_BAR''','''  if (tid == 0) {
    probe[6]=tg();
    if (atomicAdd(cnt + C_BAR''')
    source=source.replace('  // ---------------- combine:', '  if(tid==0)probe[7]=tg();\n  // ---------------- combine:')
    end=source.index('\n}\n\nsize_t smem_bytes')
    source=source[:end]+'''
  if(tid==0){probe[8]=tg();probe[18]=tg();if(!probe[4])probe[4]=probe[3];}
  __syncthreads();
  if(tid<64)trace[(size_t)c*64+tid]=probe[tid];
  __syncthreads();
  if(lane==0)trace[(size_t)c*64+40+warp]=tg();
  if(tid==0){trace[(size_t)c*64+21]=clock64();trace[(size_t)c*64+9]=tg();}
'''+source[end:]
    source=source.replace('  C10_CUDA_CHECK(cudaFuncSetAttribute(timed_moe<1>',
        '  cudaFuncAttributes a={};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,timed_moe<1>));optin-=a.sharedSizeBytes;\n  C10_CUDA_CHECK(cudaFuncSetAttribute(timed_moe<1>')
    return source


def defer_zero(source):
    a=source.index('  // zero this CTA');b=source.index('  const int q = lane & 3;',a)
    source=source[:a]+source[b:]
    source=source.replace('          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);','')
    source=source.replace('            G1 *= sc;', '            if(m==0&&i==0)ss[off+idx]=0.f;\n            G1 *= sc;')
    return source


def metadata_reader(source):
    header=(ROOT/'problems/muse_decode_layer/dev/nt1_branches/route_meta.h').read_text().replace('#pragma once','')
    source=source.replace('namespace {',header+'\nnamespace {',1)
    source=source.replace('int M, int local_start, int nstage) {','int M, int local_start, int nstage,const MoERouteMeta* meta) {')
    source=source.replace('(__nv_bfloat16*)v[12],M,start,stages);','(__nv_bfloat16*)v[12],M,start,stages,(const MoERouteMeta*)v[13]);')
    a=source.index('  for (int i = tid; i < npairs; i += NTHR)');b=source.index('  const int U1 = rt->us1[NLOC]',a)
    source=source[:a]+'''  if(tid<32){rt->n[tid]=meta->count[tid];rt->off[tid]=meta->offset[tid];rt->nblk[tid]=meta->blocks8[tid];}
  if(tid<33){rt->us1[tid]=meta->units13[tid];rt->us2[tid]=meta->units2[tid];}
  for(int i=tid;i<npairs;i+=NTHR){pos_of[i]=meta->pos_of[i];pair_of[i]=meta->pair_of[i];}
  if(tid==0){rt->c1=min(NT1,max(1,(meta->units13[32]+G/2)/G));rt->c2=min(NT2,max(1,(meta->units2[32]+G/2)/G));}
  __syncthreads();
'''+source[b:]
    a=source.index('  // ---------------- consumers: deterministic token lists');b=source.index('  // zero this CTA',a) if '  // zero this CTA' in source[a:] else source.index('  const int q = lane & 3;',a)
    source=source[:a]+source[b:]
    return source


def build():
    old=(ROOT/'prof/lab/moe_nt1_limit.py').read_text();frozen=literal(old,'NT1_SOURCES')
    z=defer_zero(frozen['lop3'])
    sources={'INC':frozen['lop3'],'TRACE':timeline(frozen['lop3']),'Z':z,'TRACEZ':timeline(z),
        'META':metadata_reader(frozen['lop3']),'METAZ':metadata_reader(z)}
    for name,source in sources.items():(DEV/(name.lower()+'.cu')).write_text(source)
    a=old.index('NT1_SOURCES=');b=old.index('\nH =',a)
    body=old[:a]+'NT1_SOURCES='+repr(sources)+old[b:old.index('\nimport hashlib\nimport statistics\nimport time\n')]
    body+='\nACTIVITY_CPP='+repr((DEV/'activity.cpp.inc').read_text())
    body+='\nCONTROL_CUDA='+repr((DEV/'controls.cu').read_text())
    body+='\n'+(DEV/'runtime.py.inc').read_text()
    base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text()
    router=select(base,{'_pdl_start','_pdl_wait','_route_fast'})
    router=router.replace('def _route_fast(', 'def timeline_route(')
    router=router.replace('X,IDS,WTS,LIDS,REPLICAS:', 'X,IDS,WTS,LIDS,META,ARRIVE,M:tl.constexpr,EMIT:tl.constexpr,REPLICAS:')
    router=re.sub(r'(?m)^def ', '@triton.jit\ndef ',router)
    body+='\nimport triton\nimport triton.language as tl\n'+router.rstrip()+'\n'+(DEV/'router_epilogue.py.inc').read_text()
    body+='\n'+(DEV/'router_runtime.py.inc').read_text()
    body+=select((ROOT/'problems/muse_decode_layer/dev/nt1_branches/fixed_runtime.py.inc').read_text(),{'nt1_fp32'})
    for path in ('diagnose.py.inc','lab.py.inc'):
        if (DEV/path).exists():body+='\n'+(DEV/path).read_text()
    out=ROOT/'prof/lab/moe_nt1_timeline.py';out.write_text(body)
    print(out.relative_to(ROOT),hashlib.sha256(body.encode()).hexdigest())


if __name__=='__main__':build()
