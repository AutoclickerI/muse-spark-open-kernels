"""Decode-once phase clocks, block-count specialization and parallel block epilogues."""
import hashlib,re
from pathlib import Path
import build_reuse as base
DEV=Path(__file__).resolve().parent
EPILOGUE=r'''
      if constexpr(MODE==2){
        float* dst=red+(NB<=2?buf*NCW*32*NB*4:0);
#pragma unroll
        for(int j=0;j<NB;++j)*reinterpret_cast<float4*>(dst+((warp*32+lane)*NB+j)*4)=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
        bar_consumers();if(tid==0)reuse_prefetch(rr,nstage,stages,fullb,emptyb,w13p,w2p);
        if(!pass){if(tid<NB*64){int i=tid&7,n=tid>>3,j=n>>3,col=n&7,ls=i*4+(col>>1),rg=col&1;float g=0.f,up=0.f;
#pragma unroll
          for(int w=0;w<NCW;++w){g+=dst[((w*32+ls)*NB+j)*4+rg];up+=dst[((w*32+ls)*NB+j)*4+rg+2];}
          if(n<ne){if(m==0&&i==0)ss[off+n]=0.f;g*=gs13[e]*256.f;up*=gs13[e]*256.f;
           inter[(size_t)(off+n)*HD+m*8+i]=__float2bfloat16(g/(1.f+__expf(-g))*up);}
        }}else{
#pragma unroll
          for(int v=0;v<(NB+1)/2;++v){int p=tid+v*256;
           if(p<NB*128){int i=p&15,n=p>>4,j=n>>3,col=n&7,ls=(i&7)*4+(col>>1),rg=(col&1)+((i>>3)<<1);float z=0.f;
#pragma unroll
            for(int w=0;w<NCW;++w)z+=dst[((w*32+ls)*NB+j)*4+rg];z*=gs2[e]*256.f;
            bool ok=n<ne;z*=1.f+__bfloat162float(pnw[m*16+i]);if(ok)y[(size_t)(off+n)*HD+m*16+i]=z;
            float sq=ok?z*z:0.f;
#pragma unroll
            for(int d=8;d;d>>=1)sq+=__shfl_xor_sync(~0u,sq,d);if(ok&&i==0)atomicAdd(ss+off+n,sq);
           }
          }
        }
        if constexpr(NB<=2)buf^=1;else bar_consumers();
      }else{
// OLD_EPILOGUE
      }
'''

def source():
 head=(DEV/'inc.cu').read_text().split('template <int NT>\n__global__')[0]
 body=(DEV/'reuse_kernel.cu.inc').read_text()
 body=body.replace('struct ReuseRoute {','struct ReuseBase {int n[32],off[32],nblk[32],us1[33],us2[33],c1,c2,run[32];PrefetchState pre;};\nstruct ReuseRoute {').replace('Route base;','ReuseBase base;')
 body=body.replace('template<bool REDO,bool TRACE>', 'template<int MODE,bool TRACE>')
 body=body.replace('uint64_t* fullb=reinterpret_cast<uint64_t*>(cached_last+8*256);','uint32_t* trace_cache=cached_last+8*256;uint64_t* fullb=reinterpret_cast<uint64_t*>(trace_cache+(TRACE?4*256:0));')
 body=body.replace('cached_b[g][7]=b.w;', 'if constexpr(TRACE){if(g<4)trace_cache[g*256+tid]=b.w;else cached_b[g][7]=b.w;}else cached_b[g][7]=b.w;')
 body=body.replace('Route* rt=&rr->base;', 'auto* rt=&rr->base;int(*wc)[32]=reinterpret_cast<int(*)[32]>(red);')
 body=body.replace('rt->wc','wc')
 body=body.replace('red+2*NCW*32*4','red+(MODE==2?4:2)*NCW*32*4')
 body=body.replace('uint64_t clocks[14]={};if constexpr(TRACE){if(tid==0)clocks[0]=reuse_clock();}', 'if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE);')
 body=re.sub(r'if constexpr\(TRACE\)\{if\(tid==0\)t=reuse_clock\(\);\}', '', body)
 body=body.replace('if constexpr(TRACE){if(tid==0)clocks[1]=reuse_clock()-clocks[0];}', 'if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+1);')
 body=body.replace('if constexpr(TRACE)clocks[6]+=reuse_clock()-t;', '')
 body=body.replace('if constexpr(TRACE){if(tid==0)clocks[7]+=reuse_clock()-t;}', '')
 body=body.replace('if constexpr(TRACE){if(tid==0)clocks[2]+=reuse_clock()-t;}', 'if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+4);')
 body=body.replace('if constexpr(TRACE){if(tid==0){clocks[5]+=reuse_clock()-t;clocks[10+pass]++;clocks[12+pass]+=nb;}}', '')
 body=body.replace('uint64_t tb=0;if constexpr(TRACE){if(tid==0)tb=reuse_clock();}', '')
 body=body.replace('bar_consumers();if constexpr(TRACE){if(tid==0)clocks[8]=reuse_clock()-tb;}', 'bar_consumers();if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+4);')
 body=body.replace('if constexpr(TRACE){if(tid==0){clocks[9]=reuse_clock();for(int k=0;k<14;++k)trace[c*16+k]=clocks[k];}}', 'if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+5);')
 body=body.replace('const int nb=rt->nblk[e],ne=rt->n[e],off=rt->off[e];','''const int nb=rt->nblk[e],ne=rt->n[e],off=rt->off[e];int seq=(pass?rr->first:0)+u-ua;
      if constexpr(TRACE)if(tid==0){phase_stamp(trace+c*PHASE_STRIDE+16+seq*8);trace[c*PHASE_STRIDE+16+seq*8+7]=(uint64_t)nb|((uint64_t)pass<<8)|((uint64_t)e<<16)|((uint64_t)m<<24);}''')
 body=body.replace('publish();cur_e=e;cur_pass=pass;seg_tiles=0;', 'publish();if constexpr(MODE==2)bar_consumers();cur_e=e;cur_pass=pass;seg_tiles=0;')
 body=body.replace('        const int idx=lane>>2;', '        if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+1);\n        const int idx=lane>>2;')
 body=body.replace('      mbar_wait(&fullb[st],ph);', '      if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+3);\n      mbar_wait(&fullb[st],ph);')
 # Two early clocks bracket the only activation-cache fill on entering an expert/pass.
 body=body.replace('      uint64_t t=0;\n', '      if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+2);\n')
 # Move the entire compute/reduce lifetime into block-count-specialized bodies, avoiding groupwise nb tests.
 a=body.index('      float acc[MAX_BLOCKS][4];');b=body.index('      ++seg_tiles;',a)
 tile=body[a:b]
 tile=tile.replace('MAX_BLOCKS','NB').replace('if(j<nb)', 'if(MODE!=0||j<nb)')
 aepi=tile.index('#pragma unroll\n      for(int j=0;j<NB;++j)if(MODE!=0||j<nb){',tile.index('// REUSE_MATH'))
 oldepi=tile[aepi:]
 tile=tile[:aepi]+EPILOGUE.replace('// OLD_EPILOGUE',oldepi)
 # Generic store stamp is after all MMA work, before warp retirement/reduce.
 tile=tile.replace('      __syncwarp();', '      if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+5);\n      __syncwarp();')
 tile+='      if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+6);\n'
 body=body[:a]+'''      auto tile_body=[&]<int NB>(){
'''+tile+'''      };
      if constexpr(MODE==0)tile_body.template operator()<4>();
      else if(nb==1)tile_body.template operator()<1>();else if(nb==2)tile_body.template operator()<2>();
      else if(nb==3)tile_body.template operator()<3>();else tile_body.template operator()<4>();
'''+body[b:]
 # Phase boundary clocks: route, W13 pass end, W2 pass end, final barrier end, finalizer end.
 body=body.replace('    }\n  }\n  publish();', '''    }
    if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+2+pass);
  }
  publish();''')
 body=body.replace('size_t reuse_smem(int nstage,int M)', 'template<int MODE,bool TRACE>size_t reuse_smem(int nstage,int M)')
 body=body.replace('(2*NCW*32*4+8*256)', '((MODE==2?4:2)*NCW*32*4+8*256+(TRACE?4*256:0))')
 body=body.replace('reuse_smem(stages,M)', '(redo==2?(trace?reuse_smem<2,true>(stages,M):reuse_smem<2,false>(stages,M)):(trace?reuse_smem<0,true>(stages,M):reuse_smem<0,false>(stages,M)))')
 body=body.replace('#define REUSE_INST(X) X(false,false) X(true,false) X(false,true) X(true,true)', '#define REUSE_INST(X) X(0,false) X(2,false) X(0,true) X(2,true)')
 math='''
      const uint4* wq=reinterpret_cast<const uint4*>(tile)+warp*8*32+lane;
      const uint4 scv=*(reinterpret_cast<const uint4*>(tile+TILE_W_BYTES)+warp*32+lane);
      const uint32_t scw[4]={scv.x,scv.y,scv.z,scv.w};
#pragma unroll
      for(int g=0;g<8;++g){
'''+base.DECODE+'''
#pragma unroll
       for(int j=0;j<NB;++j)if(MODE!=0||j<nb){
'''+base.MMA+'''
       }
      }
'''
 math=math.replace('cached_b[g][s*2+1]', '(TRACE&&g<4&&s==3?trace_cache[g*256+tid]:cached_b[g][s*2+1])')
 body=body.replace('      // REUSE_MATH',math)
 assert 'clocks[' not in re.sub(r'//[^\n]*','',body)
 helper='''constexpr int PHASE_STRIDE=16+128*8;
__device__ __forceinline__ void phase_stamp(uint64_t*p){asm volatile("{.reg .b64 t; mov.u64 t,%%globaltimer;st.global.cs.u64 [%0],t;}"::"l"(p):"memory");}
'''
 cuda=head+helper+body
 cell=base.cell_source();a=cell.index('REUSE_CUDA = ');b=cell.index('\nREUSE_CPP',a)
 cell=cell[:a]+'REUSE_CUDA = '+repr(cuda)+cell[b:]
 cell=cell.replace('for mode in (0,1)', 'for mode in (0,2)')
 cell=cell.replace('stamps.shape!=(grid,16)', 'stamps.shape!=(grid,1040)').replace('[grid,16]','[grid,1040]')
 cell=cell.replace('reuse_','reusep_').replace('REUSE_','REUSEP_');cuda=cuda.replace('reuse_','reusep_').replace('REUSE_','REUSEP_')
 return cell,cuda

def build():
 body,cuda=source()
 for p,s in ((DEV/'reusep_cell.py',body),(DEV/'reusep.cu',cuda)):
  p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
