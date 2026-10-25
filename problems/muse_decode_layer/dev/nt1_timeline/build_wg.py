"""P3a structural rebuild: native RS WGMMA on S11 packed weights, no K-split reduction or per-block re-decode."""
import ast,hashlib
from pathlib import Path
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def cuda_source():
 inc=(DEV/'inc.cu').read_text();head=inc[:inc.index('struct PrefetchState')]
 t=ast.parse((ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text())
 nf=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NF_CUDA')
 a=nf.index('template<int NT>__device__ __forceinline__ void mma(');b=nf.index('template<int NT,int VEC>',a)
 return head+(DEV/'wg_kernel.cu.inc').read_text().replace('// WG_MMA_HELPERS',nf[a:b])

def cell_source():
 return '''"""P3a Hopper RS WGMMA cell: 64 output rows, adaptive 8/16/32 expert-token columns.
prepare wg_bind(s4_nt1_bank) only reorders the existing scale bytes; all decoding runs in the timed kernel.
sm120 uses the tested NF portable mma.sync stand-in with identical stage layouts and fragment ownership.
"""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
WG_CUDA = '''+repr(cuda_source())+'''
WG_CPP = r"""#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
int64_t wg_init();std::vector<int64_t> wg_info(int64_t,int64_t);
void wg_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);
"""
_WG_MODULE=[]
def wg_module():
 if not _WG_MODULE:
  arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
  mod=load_inline(name='nt1_wg_'+hashlib.sha256(WG_CUDA.encode()).hexdigest()[:12],cpp_sources=WG_CPP,cuda_sources=WG_CUDA,
   functions=['wg_init','wg_info','wg_run'],no_implicit_headers=True,with_pytorch_error_handling=False,
   extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
  mod.optin=mod.wg_init();_WG_MODULE.append(mod)
 return _WG_MODULE[0]

def wg_bind(state):
 wg_module();bank=state['banks']['lop3'];dev=state['gs13'].device
 scales={}
 for name in ('w13p','w2p'):
  src=bank[name];ntile=src.shape[1]
  # Two u8 scale codes per lane/K64 group, still raw bytes. Adjacent lanes share one aligned 4-byte copy, selecting their own scale pair.
  raw=src[:,:,32768:].contiguous().view(32,ntile,8,32,8,2).permute(0,1,2,4,3,5).contiguous().flatten()
  scales[name]=raw
 state['wg_scales']=scales;state['wg_counters']={(s,g):torch.zeros(64,device=dev,dtype=torch.int32) for s in (4,8) for g in (2,4)}
 return state

def wg_geometry(state,m,stages=8,grid_mult=4):
 resident,regs,local,smem=wg_module().wg_info(stages,m)
 if not resident:raise ValueError('wg no resident CTA')
 return state['sms']*min(grid_mult,resident),regs,local,smem,resident

def wg_experts(data,state,stages=8,grid_mult=4):
 x=data['hidden_states'].contiguous();m=x.shape[0];p=m*8;dev=x.device
 if m>32:raise ValueError('wg accepts at most 32 tokens')
 ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();bank=state['banks']['lop3'];sf=state['wg_scales']
 inter=torch.empty((p,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((p,4096),device=dev);ss=torch.empty(p,device=dev);out=torch.empty_like(x)
 grid,*_=wg_geometry(state,m,stages,grid_mult)
 ts=[x,ids,tw,state['pnw'],bank['w13p'],bank['w2p'],sf['w13p'],sf['w2p'],state['gs13'],state['gs2'],inter,y,ss,state['wg_counters'][stages,grid_mult],out]
 wg_module().wg_run([v.data_ptr() for v in ts],m,data['config']['local_start'],stages,grid,torch.cuda.current_stream(dev).cuda_stream)
 return out
'''
def build():
 for path,src in ((DEV/'wg.cu',cuda_source()),(DEV/'wg_cell.py',cell_source())):
  path.write_text(src);print(path.name,hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
