"""Freeze exact S11, trivial reinterpret diagnostic, matched two-stage TMA loads and maximum-MLP streams."""
import ast,hashlib
from pathlib import Path
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]

def build():
 s11=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text();assert hashlib.sha256(s11.encode()).hexdigest().startswith('acda6c62')
 a=s11.index('SMALLM_VAR_CUDA = ');b=s11.index('def _s4_prepare_banks(',a);cell=s11[a:b]
 tree=ast.parse(cell);old=ast.literal_eval(tree.body[0].value)
 cuda=old
 p=cuda.index('      uint32_t o[16];',cuda.index('const uint4 scv'));q=cuda.index('#pragma unroll\n          for (int j = 0; j < NT;',p)
 # Retain all shared packed/scale loads and all MMA operands/counts. Reinterpret four words into bounded BF16s.
 cuda=cuda[:p]+'''      asm volatile(""::"r"(wv.x),"r"(wv.y),"r"(wv.z),"r"(wv.w),"r"(sw):"memory");
#pragma unroll
      for(int s=0;s<4;++s){
        const uint32_t a0=0x3e003e00u|(wv.x&0x007f007fu),a1=0x3e003e00u|(wv.y&0x007f007fu);
        const uint32_t a2=0x3e003e00u|(wv.z&0x007f007fu),a3=0x3e003e00u|(wv.w&0x007f007fu);
'''+cuda[q:]
 trivial=cell.replace(repr(old),repr(cuda),1).replace('smallm_','trivial_').replace('SMALLM','TRIVIAL')
 inc=(DEV/'inc.cu').read_text();head=inc[:inc.index('template <int NT>\n__global__')]
 loads=head+(DEV/'limit_loads.cu.inc').read_text()
 (DEV/'limit_b.cu').write_text(cuda);(DEV/'limit_c.cu').write_text(loads)
 src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text()+'\n'+cell+'\n'+trivial
 src+='\nLIMIT_C_CUDA='+repr(loads)+'\n'+(DEV/'limit_lab.py.inc').read_text()
 out=ROOT/'prof/lab/moe_nt1_limit4.py';out.write_text(src);ast.parse(src)
 for n,s in ((out,src),(DEV/'limit_b.cu',cuda),(DEV/'limit_c.cu',loads)):print(n.relative_to(ROOT),hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':build()
