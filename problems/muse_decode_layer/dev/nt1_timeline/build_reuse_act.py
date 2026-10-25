"""Overlap extra-token-block activation gathers with dequant; retain 8-way K split and2CTAs/SM."""
import hashlib
from pathlib import Path
from build_reuse_phase import source as phase_source
import build_reuse as base
DEV=Path(__file__).resolve().parent

def source():
 body,cuda=phase_source();old=cuda
 cuda=cuda.replace('red+(MODE==2?4:2)*NCW*32*4','red+2*NCW*32*4')
 cuda=cuda.replace('uint64_t* fullb=reinterpret_cast<uint64_t*>(trace_cache+(TRACE?4*256:0));', 'uint32_t* act=trace_cache+(TRACE?4*256:0);uint64_t* fullb=reinterpret_cast<uint64_t*>(act+(MODE==2?4096:0));')
 cuda=cuda.replace('((MODE==2?4:2)*NCW*32*4+8*256+(TRACE?4*256:0))', '(2*NCW*32*4+8*256+(TRACE?4*256:0)+(MODE==2?4096:0))')
 cuda=cuda.replace('if constexpr(MODE==2)bar_consumers();', '')
 cuda=cuda.replace('      if constexpr(MODE==2){\n        float* dst=', '      if constexpr(false){\n        float* dst=')
 a=cuda.index('      const uint4* wq=');b=cuda.index('      if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+5);',a)
 oldmath=cuda[a:b]
 issue=r'''
      auto act_issue=[&](int g,int j,int slot){
       if(j<NB){int idx=j*8+(lane>>2);bool valid=idx<ne;
        const __nv_bfloat16* row=pass?inter+(size_t)(off+(valid?idx:0))*HD:x+(size_t)(valid?(pair_of[off+idx]>>3):0)*HD;
        const uint4* p=reinterpret_cast<const uint4*>(row+warp*512+g*64+(lane&3)*16);
        unsigned d=smem_u32(act+slot*2048+tid*4),d1=smem_u32(act+slot*2048+1024+tid*4);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(d),"l"(p),"r"(valid?16:0):"memory");
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(d1),"l"(p+1),"r"(valid?16:0):"memory");
       }
       asm volatile("cp.async.commit_group;":::"memory");
      };
'''
 math=oldmath.replace('#pragma unroll\n      for(int g=0;g<8;++g){',issue+'''#pragma unroll
      for(int g=0;g<8;++g){
       if constexpr(MODE==2&&NB>1){act_issue(g,1,0);act_issue(g,2,1);}
''')
 math=math.replace('          if(j>0&&idx<ne){', '''          if constexpr(MODE==2&&j>0){
           asm volatile("cp.async.wait_group 1;":::"memory");
           constexpr int slot=(j-1)&1;
           v0=*reinterpret_cast<const uint4*>(act+slot*2048+tid*4);
           v1=*reinterpret_cast<const uint4*>(act+slot*2048+1024+tid*4);
           act_issue(g,j+2,slot);
          }else if(j>0&&idx<ne){''')
 # j is an unrolled loop induction variable, not a C++ constant expression.
 math=math.replace('if constexpr(MODE==2&&j>0)', 'if(MODE==2&&j>0)').replace('constexpr int slot=(j-1)&1;', 'int slot=(j-1)&1;')
 # Drain before a new g reuses both per-thread activation slots.
 target='''       }
      }
'''
 assert math.endswith(target+'\n') or math.endswith(target),repr(math[-60:])
 idx=math.rindex(target);math=math[:idx]+'''       }
       if constexpr(MODE==2&&NB>1)asm volatile("cp.async.wait_group 0;":::"memory");
      }
'''+math[idx+len(target):]
 cuda=cuda[:a]+math+cuda[b:]
 # Four B0 operand words move to shared, freeing registers for overlapped copy addresses at2 CTAs/SM.
 cuda=cuda.replace('if constexpr(TRACE){if(g<4)trace_cache', 'if constexpr(TRACE||MODE==2){if(g<4)trace_cache')
 cuda=cuda.replace('TRACE&&g<4&&s==3', '(TRACE||MODE==2)&&g<4&&s==3')
 cuda=cuda.replace('(TRACE?4*256:0)', '((TRACE||MODE==2)?4*256:0)')
 body=body.replace(repr(old),repr(cuda),1)
 body=body.replace('reusep_','reusea_').replace('REUSEP_','REUSEA_');cuda=cuda.replace('reusep_','reusea_').replace('REUSEP_','REUSEA_')
 return body,cuda

def build():
 body,cuda=source()
 for p,s in ((DEV/'reusea_cell.py',body),(DEV/'reusea.cu',cuda)):
  p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
