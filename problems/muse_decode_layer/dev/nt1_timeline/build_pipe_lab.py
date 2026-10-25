"""Single self-contained matched-cell/pipe/dependency/TMA-flight lab, synthetic data only."""
import ast,hashlib
from pathlib import Path
from build_epilogue import source as base_source
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def diagnostic():
 cell,cuda=base_source(0);old=cuda
 a=cuda.index('    {\n      const uint4* wq',cuda.index('    float acc[NT][4]'));b=cuda.index('    __syncwarp();',a)
 cuda=cuda[:a]+'''    {
      const uint4* wq=reinterpret_cast<const uint4*>(tile)+warp*8*32+lane;uint4 sv;asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(sv.x),"=r"(sv.y),"=r"(sv.z),"=r"(sv.w):"r"(smem_u32(reinterpret_cast<const uint4*>(tile+TILE_W_BYTES)+warp*32+lane)):"memory");uint32_t bs[NT]={};
#pragma unroll
      for(int g=0;g<8;++g){uint4 v;asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(smem_u32(wq+g*32)):"memory");
#pragma unroll
        for(int j=0;j<NT;++j)bs[j]^=Bf[j][g][0]^Bf[j][g][1]^Bf[j][g][2]^Bf[j][g][3]^Bf[j][g][4]^Bf[j][g][5]^Bf[j][g][6]^Bf[j][g][7]^v.x^v.y^v.z^v.w;}
#pragma unroll
      for(int j=0;j<NT;++j){float f=__uint2float_rn((bs[j]^sv.x^sv.y^sv.z^sv.w)&1023)*.00001f;
        for(int k=0;k<4;++k)acc[j][k]=f;}
    }
'''+cuda[b:]
 return cell.replace(repr(old),repr(cuda),1).replace('epib','pdia').replace('EPIB','PDIA'),cuda

def phase_cell():
 cell,cuda=base_source(0);old=cuda
 a=cuda.index('    {\n      const uint4* wq',cuda.index('    float acc[NT][4]'))
 cuda=cuda[:a]+'    uint64_t math0=0;if constexpr(TRACE)if(tid==0)math0=tg();\n'+cuda[a:]
 a=cuda.index('    __syncwarp();',a)
 cuda=cuda[:a]+'    if constexpr(TRACE)if(tid==0)probe[62+pass]+=tg()-math0;\n'+cuda[a:]
 a=cuda.index('#pragma unroll\n      for (int j = 0; j < NT; ++j)',cuda.index('  auto unit_body'))
 cuda=cuda[:a]+'      uint64_t load0=0;if constexpr(TRACE)if(tid==0)load0=tg();\n'+cuda[a:]
 a=cuda.index('    if constexpr (TRACE) { if (tid == 0 && pass == 1 && firstp)')
 assert cuda[a-6:a]=='    }\n'
 cuda=cuda[:a-6]+'      if constexpr(TRACE)if(tid==0)probe[61]+=tg()-load0;\n'+cuda[a-6:]
 return cell.replace(repr(old),repr(cuda),1).replace('epib','pclk').replace('EPIB','PCLK'),cuda

def build():
 s=(ROOT/'prof/lab/moe_nt1_reuse.py').read_text();base=ast.parse((ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text())
 cuda=next(ast.literal_eval(n.value) for n in base.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SMALLM_VAR_CUDA')
 head=cuda[:cuda.index('enum {\n  F_Z')];head+='\n__device__ __forceinline__ uint64_t tg(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t;}\n__device__ __forceinline__ uint64_t tclk(){uint64_t t;asm volatile("mov.u64 %0, %%clock64;":"=l"(t)::"memory");return t;}\n';micro=head+(DEV/'pipe_micro.cu.inc').read_text()+'\n} // namespace\n'
 # Host APIs must have external linkage.
 a=micro.index('int device_smem()');micro=micro[:a]+'} // namespace\n'+micro[a:];micro=micro.rsplit('\n} // namespace',1)[0]
 cell,diag=diagnostic();s+='\n'+cell+'\n'+phase_cell()[0]
 s+='\nPIPE_MICRO_CUDA='+repr(micro)+'\n'+(DEV/'pipe_lab.py.inc').read_text()
 ast.parse(s);p=ROOT/'prof/lab/moe_nt1_pipe.py';p.write_text(s);(DEV/'pipe_micro.cu').write_text(micro);(DEV/'pdia.cu').write_text(diag);(DEV/'pclk.cu').write_text(phase_cell()[1])
 print(p,hashlib.sha256(s.encode()).hexdigest()[:8])
if __name__=='__main__':build()
