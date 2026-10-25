"""Fixed-ownership ONCE contrast: identical math, vector reduction and separate TMA producer warp."""
import hashlib
from pathlib import Path
from build_reuse_phase import source as phase_source
from build_epilogue import VEC
DEV=Path(__file__).resolve().parent

def source():
 cell,cuda=phase_source();old=cuda
 cuda=cuda.replace('red+(MODE==2?4:2)*NCW*32*4','red+2*NCW*32*4').replace('(MODE==2?4:2)*NCW*32*4','2*NCW*32*4')
 cuda=cuda.replace('if constexpr(MODE==2)bar_consumers();','')
 cuda=cuda.replace('if(MODE!=0||j<nb)', 'if(j<nb)')
 a=cuda.index('      if constexpr(MODE==2){\n        float* dst=');b=cuda.index('      if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+6);',a)
 orig=(DEV/'reuse_kernel.cu.inc').read_text();x=orig.index('        if(!pass){if(tid<64)');y=orig.index('        buf^=1;',x);oldscalar=orig[x:y]
 vec=VEC.replace('idx=nb*8+n','idx=j*8+n')
 ep='''
#pragma unroll
      for(int j=0;j<NB;++j)if(j<nb){
       *reinterpret_cast<float4*>(red+((buf*NCW+warp)*32+lane)*4)=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
       bar_consumers();if(j==0&&tid==(MODE==2?224:0))reusep_prefetch(rr,nstage,stages,fullb,emptyb,w13p,w2p);
       const float* rb=red+buf*NCW*32*4;
       if constexpr(MODE==2){constexpr bool Z=true,ZOUT=true,VNR=false;float nv0=0,nv1=0;
'''+vec+'''       }else{
'''+oldscalar+'''       }
       buf^=1;
      }
'''
 cuda=cuda[:a]+ep+cuda[b:]
 a=cuda.index('      if constexpr(MODE==0)tile_body.template operator()<4>();');b=cuda.index('      ++seg_tiles;',a)
 cuda=cuda[:a]+'      tile_body.template operator()<4>();\n'+cuda[b:]
 cell=cell.replace(repr(old),repr(cuda),1).replace('reusep_','reusee_').replace('REUSEP_','REUSEE_')
 cuda=cuda.replace('reusep_','reusee_').replace('REUSEP_','REUSEE_')
 return cell,cuda
if __name__=='__main__':
 cell,cuda=source()
 for p,s in ((DEV/'reusee_cell.py',cell),(DEV/'reusee.cu',cuda)):
  p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
