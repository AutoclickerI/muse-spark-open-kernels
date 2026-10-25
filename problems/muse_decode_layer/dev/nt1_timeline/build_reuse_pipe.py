"""Decode once into transient shared A, overlap activation copies, parallelize the block epilogues."""
import hashlib
from pathlib import Path
from build_reuse_act import source as act_source
DEV=Path(__file__).resolve().parent

def source():
 body,cuda=act_source();old=cuda
 # On the optimized path B0 is fully cached in64 regs; shared A reuses the16KiB reduction workspace.
 cuda=cuda.replace('uint32_t cached_b[7][8];','uint32_t cached_b[8][8];')
 cuda=cuda.replace('if(g<7){cached_b[g][0]', 'if(g<7||MODE==2){cached_b[g][0]')
 cuda=cuda.replace('if constexpr(TRACE||MODE==2)', 'if constexpr(TRACE)').replace('(TRACE||MODE==2)&&g<4&&s==3','TRACE&&g<4&&s==3')
 cuda=cuda.replace('((TRACE||MODE==2)?4*256:0)', '(TRACE?4*256:0)')
 cuda=cuda.replace('g<7?', '(MODE==2||g<7)?')
 cuda=cuda.replace('red+2*NCW*32*4','red+(MODE==2?4:2)*NCW*32*4')
 cuda=cuda.replace('uint32_t* trace_cache=cached_last+8*256;', 'uint32_t* trace_cache=cached_last+(MODE==2?0:8*256);')
 cuda=cuda.replace('(2*NCW*32*4+8*256+(TRACE?4*256:0)+(MODE==2?4096:0))', '((MODE==2?4:2)*NCW*32*4+(MODE==2?0:8*256)+(TRACE?4*256:0)+(MODE==2?4096:0))')
 a=cuda.index('      const uint4* wq=');b=cuda.index('      if constexpr(TRACE)if(tid==0)phase_stamp(trace+c*PHASE_STRIDE+16+seq*8+5);',a)
 math=cuda[a:b]
 pattern='#pragma unroll\n       for(int j=0;j<NB;++j)if(MODE!=0||j<nb){'
 assert math.count(pattern)==1
 math=math.replace(pattern,'''       if constexpr(MODE==2&&NB>1){
#pragma unroll
        for(int s=0;s<4;++s){unsigned dst=smem_u32(red+s*1024+tid*4);
         asm volatile("st.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(dst),"r"(ar[s*4]),"r"(ar[s*4+1]),"r"(ar[s*4+2]),"r"(ar[s*4+3]):"memory");}
       }
'''+pattern)
 begin=math.index('          for(int s=0;s<4;++s)mma16816(');end=math.index(');',begin)+2
 call=math[begin:end];args=call[call.index('mma16816('):]
 args=args.replace('ar[s*4],ar[s*4+1],ar[s*4+2],ar[s*4+3]', 'a0,a1,a2,a3')
 new='''          for(int s=0;s<4;++s){uint32_t a0,a1,a2,a3;
           if constexpr(MODE==2&&NB>1){unsigned src=smem_u32(red+s*1024+tid*4);
            asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3):"r"(src):"memory");
           }else{a0=ar[s*4];a1=ar[s*4+1];a2=ar[s*4+2];a3=ar[s*4+3];}
           '''+args+'\n          }'
 math=math[:begin]+new+math[end:]
 cuda=cuda[:a]+math+cuda[b:]
 cuda=cuda.replace('if constexpr(false){\n        float* dst=red+(NB<=2?buf*NCW*32*NB*4:0);','if constexpr(MODE==2){\n        float* dst=red;')
 cuda=cuda.replace('if constexpr(NB<=2)buf^=1;else bar_consumers();', 'bar_consumers();')
 cuda=cuda.replace('if constexpr(MODE==2){\n        float* dst=red;', 'if constexpr(MODE==2){\n        bar_consumers();float* dst=red;')
 body=body.replace(repr(old),repr(cuda),1).replace('reusea_','reused_').replace('REUSEA_','REUSED_')
 cuda=cuda.replace('reusea_','reused_').replace('REUSEA_','REUSED_')
 return body,cuda

def build():
 body,cuda=source()
 for p,s in ((DEV/'reused_cell.py',body),(DEV/'reused.cu',cuda)):
  p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
