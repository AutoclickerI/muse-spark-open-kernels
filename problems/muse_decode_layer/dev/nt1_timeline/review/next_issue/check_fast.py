"""Execute the exact bounded FP32 normalizer and full-MoE scale extremes on GPU0."""
import hashlib,json,linecache,sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(D));import build
source='#include <cuda_runtime.h>\n#include <cstdint>\n'+build.FAST_HELPER+'''
__global__ void check(const float* meta,const float* sums,float* output){int e=threadIdx.x;if(e<32)output[e]=cancel_norm(meta,e,sums[e]);}
void run(int64_t m,int64_t s,int64_t o,int64_t st){check<<<1,32,0,(cudaStream_t)st>>>((float*)m,(float*)s,(float*)o);}
'''
mod=load_inline(name='p3a_fast_ext_'+hashlib.sha256(source.encode()).hexdigest()[:12],cpp_sources='#include <cstdint>\n#include <pybind11/pybind11.h>\nvoid run(int64_t,int64_t,int64_t,int64_t);',cuda_sources=source,functions=['run'],extra_cuda_cflags=['-O3','-arch=sm_120'],no_implicit_headers=True,with_pytorch_error_handling=False)
def metadata(scales):
 state={'gs2':scales};n={'torch':torch,'state':state};exec(build.FAST_BIND.strip().replace('\n    ','\n'),n);return n['state']['cancel_meta']
scales=torch.tensor([0.,-0.,1.,-1.,1e-40,-1e-40,1e-30,-1e-30,1e-20,-1e-20,1e-10,-1e-10,1e10,-1e10,1e20,-1e20,1e30,-1e30,1e38,-1e38,1e-12,-1e-12,1e-6,-1e-6,3.,-3.,1e-25,-1e-25,1e25,-1e25,1e-35,-1e-35],device='cuda',dtype=torch.float32)
meta=metadata(scales);a=scales.double()*256;out=torch.empty_like(scales);worst=0.;cases=0
for value in (0.,1e-40,1e-30,.5,1.,1e30,1e38):
 sums=torch.full_like(scales,value);mod.run(meta.data_ptr(),sums.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
 want=torch.where((a==0)|(sums==0),0.,a.sign()*torch.rsqrt(sums.double()/4096+1e-8/(a*a))).float()
 assert bool(torch.isfinite(out).all()),(value,out)
 err=float(((out-want).abs()/want.abs().clamp_min(1e-38)).max());assert err<2e-6,(value,err,out,want);worst=max(worst,err);cases+=32
# Complete native MoE under changing global scales, including cancellation's zero/sign cases.
text=(R/'prof/lab/moe_nt1_reuse.py').read_text();f='<next fixtures>';linecache.cache[f]=(len(text),None,text.splitlines(True),f);n={'__name__':'next_fixture'};exec(compile(text,f,'exec'),n)
w=n['rg_generate_weights'](5353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=n['nt1_banks'](w);state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']};original=state['gs2'].clone();m=json.loads((D/'components_manifest.json').read_text());checks=0
for arm in ('FAST0','FAST1','FAST2'):
 spec=m[arm];t=Path(R/spec['cell']).read_text();f='<next cell '+arm+'>';linecache.cache[f]=(len(t),None,t.splitlines(True),f);c={**n,'__name__':'next_'+arm};exec(compile(t,f,'exec'),c);tag=spec['prefix']
 for mm in spec['M']:
  data=n['fx_make_fixture'](w,mm,29717+mm)
  for factor in (0.,-1.,1e-10,1e10):
   state['gs2'].copy_(original*factor);w['w2_gs']=state['gs2'];c[tag+'_bind'](state,{1:95,8:69});fn=lambda:c[tag+'_experts'](data,state,95 if mm==1 else 69)
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):got=fn()
   g.replay();torch.cuda.synchronize();want=n['nt1_fp32'](data,w);ok,total,row=n['task_check'](got,want);assert ok,(arm,mm,factor,total,row);checks+=1
 state['gs2'].copy_(original);w['w2_gs']=state['gs2']
result=dict(helper_sha256=hashlib.sha256(build.FAST_HELPER.encode()).hexdigest(),helper_cases=cases,helper_worst_relative=worst,complete_MoE_extreme_TASK_checks=checks,metadata='FP32 float4: min(1,q²),4096*min(1,1/q²),64*sign(q)*min(1,abs(q)),original_gs2; q=256*gs2/sqrt(eps)',timed_math='sum==0?0:num*rsqrt(fma(sum,mul,eps)); no FP64,sign or exponent branch')
(D/'fast_extremes.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
