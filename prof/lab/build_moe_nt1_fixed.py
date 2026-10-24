"""Build composable routing, expert-ready, W2 lookahead and warp-pipeline experiments."""
import hashlib
from pathlib import Path

from build_moe_nt1_limit import literal
from build_moe_small_debug import select

ROOT=Path(__file__).resolve().parents[2]
DEV=ROOT/'problems/muse_decode_layer/dev/nt1_branches'


def fixed(source,a=False,b=False,c=False):
    flags=f'\n#define META_ON {int(a)}\n#define READY_ON {int(b)}\n#define W2_AHEAD {int(c)}\n'
    source=source.replace('namespace {',flags+(DEV/'route_meta.h').read_text()+'\nnamespace {',1)
    source=source.replace('int M, int local_start, int nstage) {','int M, int local_start, int nstage, const MoERouteMeta* meta) {')
    source=source.replace('(__nv_bfloat16*)v[12],M,start,stages);','(__nv_bfloat16*)v[12],M,start,stages,(const MoERouteMeta*)v[13]);')
    p=source.index('template <int NT>\n__global__')
    source=source[:p]+(DEV/'route_meta.cu.inc').read_text()+'\n'+(DEV/'ready.cu.inc').read_text()+'\n'+source[p:]
    p=source.index('  for (int i = tid; i < npairs; i += NTHR)')
    q=source.index('  const int U1 = rt->us1[NLOC]',p)
    old=source[p:q]
    source=source[:p]+'''#if META_ON
  if(tid<32){rt->n[tid]=meta->count[tid];rt->off[tid]=meta->offset[tid];rt->nblk[tid]=meta->blocks8[tid];}
  if(tid<33){rt->us1[tid]=meta->units13[tid];rt->us2[tid]=meta->units2[tid];}
  for(int i=tid;i<npairs;i+=NTHR){pos_of[i]=meta->pos_of[i];pair_of[i]=meta->pair_of[i];}
  if(tid==0){rt->c1=min(NT1,max(1,(meta->units13[32]+G/2)/G));rt->c2=min(NT2,max(1,(meta->units2[32]+G/2)/G));}
  __syncthreads();
#else
'''+old+'#endif\n'+source[q:]
    p=source.index('  // ---------------- consumers: deterministic token lists')
    q=source.index('  // zero this CTA',p)
    source=source[:p]+'#if !META_ON\n'+source[p:q]+'#endif\n'+source[q:]
    p=source.index('  // zero this CTA');q=source.index('  const int q = lane & 3;',p)
    source=source[:p]+'''#if READY_ON
  // Remote-only tokens have no expert finalizer.
  if(c<M){int n=0;
    #pragma unroll
    for(int s=0;s<8;++s){int e0=ids[c*8+s]-local_start;n+=(e0>=0&&e0<32);}
    if(n==0)for(int col=tid;col<HD;col+=NTHR)out[(size_t)c*HD+col]=__float2bfloat16(0.f);
  }
#else
'''+source[p:q]+'''#endif
#if W2_AHEAD
  // Timed lookahead of assigned W2 cache lines while W13 runs. Same bank.
  for(int u=a2;u<b2;++u){
    int pe,pnb,pm;decode_unit(u,rt->us2,rt->nblk,rt->c2,NT2,pe,pnb,pm);
    const uint8_t* src=w2p+((size_t)pe*256+pm)*36864;
    for(int byte=tid*128;byte<36864;byte+=NTHR*128)
      asm volatile("prefetch.global.L2 [%0];"::"l"(src+byte));
  }
#endif
'''+source[q:]
    old='''        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment
          __threadfence();
          bar_consumers();
          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);
        }'''
    assert old in source
    source=source.replace(old,'''#if READY_ON
        if(cur_pass>=0&&seg_tiles>0)close_segment(cur_pass,cur_e,seg_tiles,rt,cnt,y,ss,pnw,ids,tw,pos_of,pair_of,out,M,local_start);
#else
'''+old+'\n#endif')
    source=source.replace('          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);',
                         '#if !READY_ON\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n#endif')
    source=source.replace('            G1 *= sc;', '''#if READY_ON
            if(m==0&&i==0)ss[off+idx]=0.f;
#endif
            G1 *= sc;''')
    p=source.rindex('  if (cur_pass == 0 && seg_tiles > 0)');q=source.index('\n}\n\nsize_t smem_bytes',p)
    source=source[:p]+'''#if READY_ON
  if(cur_pass>=0&&seg_tiles>0)close_segment(cur_pass,cur_e,seg_tiles,rt,cnt,y,ss,pnw,ids,tw,pos_of,pair_of,out,M,local_start);
#else
'''+source[p:q]+'\n#endif'+source[q:]
    source=source.replace(' moe_fused<1><<<grid,NTHR,', '''#if META_ON
 make_route_meta<<<1,256,0,(cudaStream_t)stream>>>((const int*)v[1],M,start,(MoERouteMeta*)v[13]);
#endif
 moe_fused<1><<<grid,NTHR,''')
    return source


def warp_split(source,chunk=1):
    source=source.replace('constexpr int TILE_BYTES = 36864;',f'constexpr int WS_SUB={chunk},WS_RAW=4608*WS_SUB,WS_WORDS=4096*WS_SUB,WS_GROUPS=8/WS_SUB;\nconstexpr int TILE_BYTES=WS_RAW;')
    source=source.replace('constexpr int NTHR = NCT;', 'constexpr int NTHR = NCT+128;')
    source=source.replace('__launch_bounds__(NTHR, 2)','__launch_bounds__(NTHR, 2)')
    source=source.replace('float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);',
        'uint32_t* decoded=reinterpret_cast<uint32_t*>(smem+nstage*TILE_BYTES);\n  float* red = reinterpret_cast<float*>(decoded+nstage*WS_WORDS);')
    source=source.replace('Route* rt = reinterpret_cast<Route*>(emptyb + nstage);',
        'uint64_t* dfull=emptyb+nstage;uint64_t* dempty=dfull+nstage;\n  Route* rt = reinterpret_cast<Route*>(dempty+nstage);')
    source=source.replace('mbar_init(&emptyb[s], NCW);','mbar_init(&emptyb[s],4);mbar_init(&dfull[s],4);mbar_init(&dempty[s],NCW);')
    source=source.replace('''    for (int i=0;i<nstage;++i)
      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);''','')
    p=source.index('  // ---------------- consumers: deterministic token lists');q=source.index('  // zero this CTA',p)
    # Restrict the old 256-thread deterministic-sort named barriers.
    source=source[:p]+'  if(tid<NCT){\n'+source[p:q]+'  }\n'+source[q:]
    p=source.index('  // zero this CTA');q=source.index('#endif\n#if W2_AHEAD',p)
    source=source[:p]+'  if(tid<NCT){\n'+source[p:q]+'  }\n'+source[q:]
    p=source.index('  const int q = lane & 3;')
    source=source[:p]+'''  __syncthreads();
  if(tid>=NCT){
    produce_ws(tid-NCT,rt,total_count,first_count,a1,a2,nstage,stages,decoded,fullb,emptyb,dfull,dempty,w13p,w2p);
    return;
  }
'''+source[p:]
    source=source.replace('  uint32_t Bf[NT][8][8];','')
    p=source.index('#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB')
    q=source.index('      if (pass == 1 && tid < 16 * NB)',p)
    source=source[:p]+'      }\n'+source[q:]
    p=source.index('      mbar_wait(&fullb[st], ph);');q=source.index('      float* rp = red',p)
    source=source[:p]+(DEV/'ws_consume.cu.inc').read_text()+'\n'+source[q:]
    source=source.replace('      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);','')
    source=source.replace('static_cast<size_t>(nstage) * TILE_BYTES +',
                         'static_cast<size_t>(nstage) * (TILE_BYTES+WS_WORDS*sizeof(uint32_t)) +')
    source=source.replace('2 * nstage * 8 +','4 * nstage * 8 +')
    p=source.index('template <int NT>\n__global__')
    source=source[:p]+(DEV/'ws_produce.cu.inc').read_text()+'\n'+source[p:]
    return source


def build():
    old=(ROOT/'prof/lab/moe_nt1_limit.py').read_text();frozen=literal(old,'NT1_SOURCES')
    sources={'R2':frozen['recode'],'INC':frozen['lop3']}
    sources['L26']=(DEV/'lop26.cu').read_text();sources['K2']=(DEV/'split2.cu').read_text()
    for name,a,b,c in [('A',1,0,0),('B',0,1,0),('C',0,0,1),('ABC',1,1,1)]:
        sources[name]=fixed(frozen['lop3'],a,b,c)
    sources['D']=warp_split(fixed(frozen['lop3']))
    sources['ABCD']=warp_split(fixed(frozen['lop3'],True,True,True))
    sources['D128']=warp_split(fixed(frozen['lop3']),2)
    sources['ABCD128']=warp_split(fixed(frozen['lop3'],True,True,True),2)
    for name,source in sources.items():(DEV/('fixed_'+name+'.cu')).write_text(source)
    a=old.index('NT1_SOURCES=');b=old.index('\nH =',a)
    text=old[:a]+'NT1_SOURCES='+repr(sources)+old[b:old.index('\nimport hashlib\nimport statistics\nimport time\n')]
    text+='\n'+(DEV/'fixed_runtime.py.inc').read_text()+'\n'+(DEV/'fixed_lab.py.inc').read_text()
    out=ROOT/'prof/lab/moe_nt1_fixed.py';out.write_text(text)
    helpers=select(old,{'H','NLOC','TOPK','TILE_BYTES','NT1','NCNT','_DQ_SLOTS','_LA_MASK','_bit_table','_TABLE','_encode_scales','_repack'})
    runtime=select((DEV/'fixed_runtime.py.inc').read_text(),{'NT1_CPP','NT1_MODULES','NT1_HOPPER','nt1_module','nt1_call'})
    cell='import torch\nimport hashlib\nfrom torch.utils.cpp_extension import load_inline\nNT1_SOURCES='+repr(sources)+'\n'+helpers+runtime
    cell+=(DEV/'fixed_prepare.py.inc').read_text()
    (DEV/'fixed_cell.py').write_text(cell)
    print(out.relative_to(ROOT),hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':build()
