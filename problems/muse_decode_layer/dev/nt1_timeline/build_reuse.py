"""Freeze the streamed-activation decode-once cell and its directly paired S11 lab."""
from pathlib import Path
import hashlib

DEV=Path(__file__).resolve().parent
ROOT=DEV.parents[3]

DECODE='''
          const uint4 wv=wq[g*32];const uint32_t sw=scw[g>>1];
          const uint32_t Sr=__byte_perm(sw,0u,(g&1)?0x4242u:0x4040u)*16u+0x70007000u;
          const uint32_t Sr8=__byte_perm(sw,0u,(g&1)?0x4343u:0x4141u)*16u+0x70007000u;
          uint32_t o[16],ar[16];dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);
#pragma unroll
          for(int s=0;s<4;++s){int k=(s>>1)*8+(s&1)*4;
            ar[s*4]=hmul2(o[k],Sr);ar[s*4+1]=hmul2(o[k+1],Sr8);
            ar[s*4+2]=hmul2(o[k+2],Sr);ar[s*4+3]=hmul2(o[k+3],Sr8);}
'''
MMA='''
          const int idx=j*8+(lane>>2);uint4 v0=make_uint4(0,0,0,0),v1=v0;
          if(j>0&&idx<ne){const int pos=off+idx;
            const __nv_bfloat16* row=pass?inter+(size_t)pos*HD:x+(size_t)(pair_of[pos]>>3)*HD;
            const uint4* p=reinterpret_cast<const uint4*>(row+warp*512+g*64+(lane&3)*16);
            if(pass){v0=__ldcg(p);v1=__ldcg(p+1);}else{v0=__ldg(p);v1=__ldg(p+1);}}
          const uint32_t bf[8]={v0.x,v0.y,v0.z,v0.w,v1.x,v1.y,v1.z,v1.w};
#pragma unroll
          for(int s=0;s<4;++s)mma16816(acc[j],ar[s*4],ar[s*4+1],ar[s*4+2],ar[s*4+3],j==0?(g<7?cached_b[g][s*2]:cached_last[(s*2)*256+tid]):bf[s*2],j==0?(g<7?cached_b[g][s*2+1]:cached_last[(s*2+1)*256+tid]):bf[s*2+1]);
'''

def timed(code,slot):
    return ('if constexpr(TRACE){if(tid==0)t=reuse_clock();}\n'+code+
            f'if constexpr(TRACE){{if(tid==0)clocks[{slot}]+=reuse_clock()-t;}}\n')

def cuda_source():
    inc=(DEV/'inc.cu').read_text();head=inc[:inc.index('template <int NT>\n__global__')]
    math='''
      const uint4* wq=reinterpret_cast<const uint4*>(tile)+warp*8*32+lane;
      const uint4 scv=*(reinterpret_cast<const uint4*>(tile+TILE_W_BYTES)+warp*32+lane);
      const uint32_t scw[4]={scv.x,scv.y,scv.z,scv.w};
      if constexpr(REDO){
#pragma unroll
        for(int j=0;j<MAX_BLOCKS;++j)if(j<nb){
#pragma unroll
          for(int g=0;g<8;++g){
'''+timed(DECODE,3)+timed(MMA,4)+'''
          }
        }
      }else{
#pragma unroll
        for(int g=0;g<8;++g){
'''+timed(DECODE,3)+'''
#pragma unroll
          for(int j=0;j<MAX_BLOCKS;++j)if(j<nb){
'''+timed(MMA,4)+'''
          }
        }
      }
'''
    return head+(DEV/'reuse_kernel.cu.inc').read_text().replace('      // REUSE_MATH',math)


def cell_source():
    return '''"""P3a decode-once: one unique weight tile, per-K-group activation streaming over all token blocks.
Bind in prepare with reuse_bind(s4_nt1_bank); same packed bank and data ABI as the S11 small-M cell.
mode 0=ONCE, 1=REDO fixed-work diagnostic; trace=True emits per-CTA clocks without changing the untraced build.
"""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
REUSE_CUDA = '''+repr(cuda_source())+'''
REUSE_CPP = r"""#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
int64_t reuse_init();std::vector<int64_t> reuse_info(int64_t,int64_t,int64_t,int64_t);
void reuse_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
"""
_REUSE_MODULE=[]

def reuse_module():
    if not _REUSE_MODULE:
        target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
        mod=load_inline(name='muse_smallm_reuse_'+hashlib.sha256(REUSE_CUDA.encode()).hexdigest()[:12],
            cpp_sources=REUSE_CPP,cuda_sources=REUSE_CUDA,functions=['reuse_init','reuse_info','reuse_run'],
            no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
        mod.optin=mod.reuse_init();_REUSE_MODULE.append(mod)
    return _REUSE_MODULE[0]

def reuse_bind(state):
    reuse_module()
    state['reuse_counters']={mode:torch.zeros(64,device=state['gs13'].device,dtype=torch.int32) for mode in (0,1)}
    return state

def reuse_geometry(state,m,mode=0,trace=False,stages=2):
    resident,regs,local,smem=reuse_module().reuse_info(mode,int(trace),stages,m)
    if not resident:raise ValueError('reuse has no resident CTA')
    return min(264,state['sms']*resident),regs,local,smem,resident

def reuse_experts(data,state,mode=0,trace=False,stamps=None):
    x=data['hidden_states'].contiguous();m=x.shape[0];dev=x.device;p=m*8
    if m>32:raise ValueError('reuse accepts at most 32 tokens')
    stages=2;grid,*_=reuse_geometry(state,m,mode,trace,stages)
    inter=torch.empty((p,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((p,4096),device=dev);ss=torch.empty(p,device=dev)
    out=torch.empty_like(x);ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();bank=state['banks']['lop3']
    ts=[x,ids,tw,state['pnw'],bank['w13p'],bank['w2p'],state['gs13'],state['gs2'],inter,y,ss,state['reuse_counters'][mode],out]
    if trace and (stamps is None or stamps.shape!=(grid,16)):raise ValueError('reuse trace needs [grid,16] int64')
    reuse_module().reuse_run([t.data_ptr() for t in ts]+[stamps.data_ptr() if trace else 0],m,data['config']['local_start'],
        stages,grid,torch.cuda.current_stream(dev).cuda_stream,mode,int(trace))
    return out
'''

def build():
    cuda=cuda_source();(DEV/'reuse.cu').write_text(cuda)
    cell=cell_source();(DEV/'reuse_cell.py').write_text(cell)
    for name,s in (('reuse.cu',cuda),('reuse_cell.py',cell)):print(name,hashlib.sha256(s.encode()).hexdigest())

if __name__=='__main__':build()
