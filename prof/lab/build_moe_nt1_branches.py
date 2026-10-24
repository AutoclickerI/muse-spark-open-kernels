"""Prepare small-M instruction and resident-CTA experiments from frozen NT1 controls."""
import hashlib
from pathlib import Path

from build_moe_nt1_limit import literal, lop3
from build_moe_small_debug import select

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / 'problems/muse_decode_layer/dev/nt1_branches'


def split2(source):
    source = lop3(source)
    for old,new in [('TILE_BYTES = 36864','TILE_BYTES = 18432'),
                    ('TILE_W_BYTES = 32768','TILE_W_BYTES = 16384'),
                    ('NT1 = 512','NT1 = 1024'),('NT2 = 256','NT2 = 512'),
                    ('NCW = 8','NCW = 4')]:
        assert old in source;source=source.replace(old,new)
    source=source.replace('float* __restrict__ ss, int* __restrict__ cnt,',
                          'float* __restrict__ ss, float* __restrict__ parts, int* __restrict__ tickets, int* __restrict__ cnt,')
    source=source.replace('row + warp * 512 + g * 64 + q * 16',
                          'row + (m/(pass?256:512))*2048 + warp * 512 + g * 64 + q * 16')
    # Activation fragments must change when the K half changes, even within
    # one expert/token block. Publication counts include every half tile.
    source=source.replace('int cur_pass = -1,', 'int cur_split=-1; int cur_pass = -1,')
    source=source.replace('if (pass != cur_pass || e != cur_e || nb != cur_nb) {',
                          'if (pass != cur_pass || e != cur_e || nb != cur_nb || cur_split != m/(pass?256:512)) {')
    source=source.replace('cur_pass = pass; cur_e = e;', 'cur_split=m/(pass?256:512); cur_pass = pass; cur_e = e;')
    source=source.replace('pnw[m * 16 + (tid & 15)]','pnw[(m % 256) * 16 + (tid & 15)]')
    a=source.index('      if (pass == 0) {\n        const float sc = gs13[e]')
    b=source.index('      buf ^= 1;',a)
    source=source[:a]+'''
      // Split halves publish disjoint FP32 partials; the second arrival
      // reduces before SwiGLU or normalization. No activation rounding
      // occurs between split-K partials.
      const int tile_m=m%(pass?256:512),split=m/(pass?256:512);
      const int ticket=(off+nb*NB)*768+(pass?512:0)+tile_m;
      const int active=pass?16*NB:8*NB;
      if(tid<active){
        const int i=pass?(tid&15):(tid&7),n=pass?(tid>>4):(tid>>3);
        const int col=n&7,ls=(i&7)*4+(col>>1),rg=(col&1)+(pass?((i>>3)<<1):0);
        float v=0.f,uval=0.f;
        #pragma unroll
        for(int w=0;w<NCW;++w){
          v+=rb[(w*32+ls)*RED+rg];
          if(!pass)uval+=rb[(w*32+ls)*RED+rg+2];
        }
        const int idx=nb*NB+n;
        if(idx<ne){
          const size_t dst=((size_t)(off+idx)*2+split)*8192+(pass?tile_m*16+i:tile_m*8+i);
          parts[dst]=v;if(!pass)parts[dst+4096]=uval;
        }
      }
      __threadfence();bar_consumers();
      if(tid==0){
        int previous;
        asm volatile("atom.acq_rel.gpu.global.add.s32 %0,[%1],1;":"=r"(previous):"l"(tickets+ticket):"memory");
        rt->wc[0][0]=previous;
      }
      bar_consumers();
      if(rt->wc[0][0]==1){
        if(tid<active){
          const int i=pass?(tid&15):(tid&7),n=pass?(tid>>4):(tid>>3);
          const int idx=nb*NB+n;
          if(idx<ne){
            const size_t dst=(size_t)(off+idx)*2*8192+(pass?tile_m*16+i:tile_m*8+i);
            float v=__ldcg(parts+dst)+__ldcg(parts+dst+8192);
            if(!pass){
              float up=__ldcg(parts+dst+4096)+__ldcg(parts+dst+8192+4096);
              v*=gs13[e]*256.f;up*=gs13[e]*256.f;
              inter[(size_t)(off+idx)*HD+tile_m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*up);
            }else{
              v*=gs2[e]*256.f;
              y[(size_t)(off+idx)*HD+tile_m*16+i]=v;
            }
          }
        }
        // Each warp holds two tokens, each in a 16-lane segment.
        if(pass){
          const int idx=nb*NB+(tid>>4),i=tid&15;
          float sq=0.f;
          if(idx<ne){float v=__ldcg(y+(size_t)(off+idx)*HD+tile_m*16+i)*fw[0];sq=v*v;}
          #pragma unroll
          for(int d=8;d>0;d>>=1)sq+=__shfl_xor_sync(~0u,sq,d);
          if(idx<ne&&i==0)atomicAdd(ss+off+idx,sq);
        }
        bar_consumers();
        if(tid==0)atomicExch(tickets+ticket,0);
      }
      bar_consumers();
''' + source[b:]
    # The output combine processes 8 channels/thread; 128-thread CTAs cover
    # one quarter of a token instead of the baseline's half token.
    source=source.replace('it < 2 * M','it < 4 * M').replace('const int t = it >> 1;', 'const int t = it >> 2;')
    source=source.replace('(it & 1) * 2048 + tid * 8','(it & 3) * 1024 + tid * 8')
    source=source.replace('(float*)v[10],(int*)v[11]', '(float*)v[10],(float*)v[13],(int*)v[14],(int*)v[11]')
    return source


def build():
    DEV.mkdir(exist_ok=True,parents=True)
    # This file was frozen by the preceding audit, avoiding unrelated changes
    # in another lane's live candidate while a lab is pending.
    frozen=literal((ROOT/'prof/lab/moe_nt1_limit.py').read_text(),'NT1_SOURCES')
    baseline=frozen['recode'];raw=frozen['raw']
    lop=lop3(baseline).replace('__launch_bounds__(NTHR, 2)', '__maxnreg__(124)')
    sources=dict(recode=baseline,raw=raw,lop26=lop,split2=split2(baseline))
    for name,source in sources.items():(DEV/(name+'.cu')).write_text(source)
    old=(ROOT/'prof/lab/moe_nt1_limit.py').read_text()
    a=old.index('NT1_SOURCES=');b=old.index('\nH =',a)
    body=old[:a]+'NT1_SOURCES='+repr(sources)+old[b:old.index('\nimport hashlib\nimport statistics\nimport time\n')]
    body+='\nNT1_BRANCH=__BRANCH__\n'+(DEV/'driver.py.inc').read_text()
    for branch,name in [('both','moe_nt1_branches'),('decode','moe_nt1_decode_ready'),('occupancy','moe_nt1_occupancy_ready')]:
        target=ROOT/'prof/lab'/(name+'.py');text=body.replace('__BRANCH__',repr(branch));target.write_text(text)
        print(target.relative_to(ROOT),hashlib.sha256(text.encode()).hexdigest())
    helpers=select(old,{'H','NLOC','TOPK','TILE_BYTES','NT1','NCNT','_DQ_SLOTS','_LA_MASK','_bit_table','_TABLE','_encode_scales','_repack','_raw_repack'})
    driver=(DEV/'driver.py.inc').read_text()
    runtime=select(driver,{'NT1_CPP','NT1_MODULES','NT1_HOPPER','nt1_module','nt1_call'})
    drop='import torch\nimport hashlib\nfrom torch.utils.cpp_extension import load_inline\nNT1_SOURCES='+repr(sources)+'\n'+helpers+runtime
    drop+=(DEV/'prepare.py.inc').read_text()
    (DEV/'dropin.py').write_text(drop)
    return ROOT/'prof/lab/moe_nt1_branches.py'


if __name__=='__main__':build()
