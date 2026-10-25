"""Actual capture dependency/visibility probe: unchanged producer has no release instruction."""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
H=Path(__file__).resolve().parent
src='#include <cuda_runtime.h>\n#include <cstdint>\n#include <string>\n#include <stdexcept>\n'+(H/'early_edge.cu.inc').read_text()+r'''
__device__ unsigned long long tm(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__global__ void producer(int* data,unsigned long long* clk,int value){if(threadIdx.x==0){clk[0]=tm();const auto c=clock64();while(clock64()-c<350000){}*data=value;__threadfence();clk[1]=tm();}}
__global__ void consumer(const int* data,unsigned long long* clk,int* out){if(threadIdx.x==0)clk[2]=tm();asm volatile("griddepcontrol.wait;":::"memory");if(threadIdx.x==0){*out=*data;clk[3]=tm();}}
void pair(int64_t data,int64_t clock,int64_t output,int64_t value,int64_t stream,int64_t early){
 auto st=(cudaStream_t)stream;auto d=(int*)data;auto c=(unsigned long long*)clock;auto o=(int*)output;
 producer<<<1,32,0,st>>>(d,c,(int)value);
 void* args[]={&d,&c,&o};cudaKernelNodeParams p={};p.func=(void*)consumer;p.gridDim=dim3(1);p.blockDim=dim3(32);p.kernelParams=args;
 if(vr_capture_consumer(st,p,1,early))return;
 cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(1);cfg.blockDim=dim3(32);cfg.stream=st;cudaLaunchAttribute at={};at.id=cudaLaunchAttributeProgrammaticStreamSerialization;at.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&at;cfg.numAttrs=1;
 vr_check(cudaLaunchKernelEx(&cfg,consumer,d,c,o),"probe launch");
}
'''
major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
mod=load_inline(name='vr_edge_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources='#include <cstdint>\nvoid pair(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);',cuda_sources=src,functions=['pair'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
records=[]
for early in (False,True):
 data=torch.zeros(1,device='cuda',dtype=torch.int32);out=torch.zeros_like(data);clock=torch.zeros(4,device='cuda',dtype=torch.int64)
 value=torch.empty(1,device='cuda') # maintain stable allocations before graph capture
 def call():mod.pair(data.data_ptr(),clock.data_ptr(),out.data_ptr(),98731,torch.cuda.current_stream().cuda_stream,int(early))
 call();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):call()
 for i in range(32):
  data.fill_(-111-i);out.fill_(-1);g.replay();torch.cuda.synchronize();assert int(out.item())==98731
 clocks=clock.cpu().tolist();records.append(dict(early=early,producer_start=clocks[0],producer_end=clocks[1],consumer_start=clocks[2],consumer_wait_done=clocks[3],overlap_ns=clocks[1]-clocks[2],fresh_replays=32,correct=True))
 print(records[-1],flush=True)
assert records[1]['consumer_wait_done']>=records[1]['producer_end']
(H/'out/edge_admission.json').write_text(json.dumps(dict(records=records,source_sha256=hashlib.sha256(src.encode()).hexdigest()),indent=2)+'\n')
print('PASS actual launch-completion edge, dependency visibility,32fresh replays per arm; overlap is opportunistic')
