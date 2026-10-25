"""Compact route scratch below the196KiB two-CTA boundary, optionally prefetch activation operands."""
import hashlib
from pathlib import Path
from build_reuse_phase import source as phase_source
DEV=Path(__file__).resolve().parent

def source(prefetch=False):
 body,cuda=phase_source();old=cuda
 cuda=cuda.replace('nblk[32],us1[33],us2[33],c1,c2,run[32]', 'nblk[32],c1,c2,run[32]')
 cuda=cuda.replace('int(*wc)[32]=reinterpret_cast<int(*)[32]>(red);', 'int(*wc)[32]=reinterpret_cast<int(*)[32]>(red);int* us1=reinterpret_cast<int*>(red)+8*32;int* us2=us1+33;')
 kernel=cuda.index('template<int MODE,bool TRACE>')
 cuda=cuda[:kernel]+cuda[kernel:].replace('rt->us1','us1').replace('rt->us2','us2')
 if prefetch:
  a=cuda.index('      const uint4* wq=');b=cuda.index('      if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+5);',a)
  math=cuda[a:b]
  code='''
       if constexpr(MODE==2&&NB>1){
#pragma unroll
        for(int j=1;j<NB;++j){int idx=j*8+(lane>>2);
         if(idx<ne&&(lane&3)==0){const __nv_bfloat16* row=pass?inter+(size_t)(off+idx)*HD:x+(size_t)(pair_of[off+idx]>>3)*HD;
          const void* p=row+warp*512+g*64;asm volatile("prefetch.global.L1 [%0];"::"l"(p):"memory");}
        }
       }
'''
  math=math.replace('      for(int g=0;g<8;++g){', '      for(int g=0;g<8;++g){'+code)
  # Ordinary cached loads share the prefetched L1 path. The route acquire precedes all W2 input reads.
  math=math.replace('if(pass){v0=__ldcg(p);v1=__ldcg(p+1);}else{v0=__ldg(p);v1=__ldg(p+1);}', '''if constexpr(MODE==2){
             asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v0.x),"=r"(v0.y),"=r"(v0.z),"=r"(v0.w):"l"(p):"memory");
             asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v1.x),"=r"(v1.y),"=r"(v1.z),"=r"(v1.w):"l"(p+1):"memory");
            }else if(pass){v0=__ldcg(p);v1=__ldcg(p+1);}else{v0=__ldg(p);v1=__ldg(p+1);}''')
  cuda=cuda[:a]+math+cuda[b:]
 # Hopper reserves1KiB/CTA. Move one last-group operand into a register as well: total saving1288B.
 cuda=cuda.replace('uint32_t cached_b[7][8];','uint32_t cached_b[7][8];uint32_t cached_extra=0;')
 cuda=cuda.replace('cached_last+8*256','cached_last+7*256').replace('+8*256+(TRACE?4*256:0)', '+7*256+(TRACE?4*256:0)')
 cuda=cuda.replace('for(int k=0;k<8;++k)cached_last[k*256+tid]=v[k];','for(int k=0;k<8;++k){if(k==0)cached_extra=v[k];else cached_last[(k-1)*256+tid]=v[k];}')
 cuda=cuda.replace('cached_last[(s*2)*256+tid]', '(s==0?cached_extra:cached_last[(s*2-1)*256+tid])')
 cuda=cuda.replace('cached_last[(s*2+1)*256+tid]', 'cached_last[(s*2)*256+tid]')
 #85% of228KiB rounds to196KiB. The driver can override it if a traced launch needs more shared.
 cuda=cuda.replace(' REUSEP_INST(ATTR)', ' REUSEP_INST(ATTR)')
 needle='C10_CUDA_CHECK(cudaFuncSetAttribute(moe_reuse<R,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));'
 replacement=needle+'int major=0;C10_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));if(major==9&&!T)C10_CUDA_CHECK(cudaFuncSetAttribute(moe_reuse<R,T>,cudaFuncAttributePreferredSharedMemoryCarveout,85));'
 assert needle in cuda;cuda=cuda.replace(needle,replacement)
 body=body.replace(repr(old),repr(cuda),1)
 prefix='reusef_' if prefetch else 'reusec_';upper='REUSEF_' if prefetch else 'REUSEC_'
 body=body.replace('reusep_',prefix).replace('REUSEP_',upper);cuda=cuda.replace('reusep_',prefix).replace('REUSEP_',upper)
 return body,cuda

def build():
 for pre in (False,True):
  body,cuda=source(pre);tag='reusef' if pre else 'reusec'
  for p,s in ((DEV/(tag+'_cell.py'),body),(DEV/(tag+'.cu'),cuda)):
   p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
