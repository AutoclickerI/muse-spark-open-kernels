"""Generate pointer-only bindings so neither CUDA nor pybind parses PyTorch headers."""
import re
PRELUDE='''#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)
#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)
'''
CPP_PRELUDE='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\n'
FX_CPP='int64_t fx_init();std::vector<int64_t> fx_info(int64_t,int64_t,int64_t);void fx_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void fx_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void fx_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);'
PIPE_CPP='int64_t pipe_init();std::vector<int64_t> pipe_info(int64_t);void pipe_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'

def lean(s):
 s=s.replace('#include <torch/extension.h>','').replace('#include <ATen/cuda/CUDAContext.h>','')
 s=s.replace('C10_CUDA_KERNEL_LAUNCH_CHECK()','WX_CUDA_CHECK(cudaGetLastError())').replace('C10_CUDA_CHECK','WX_CUDA_CHECK').replace('TORCH_CHECK','WX_CHECK')
 return s

def fixed(s):
 s=lean(s);a=s.index('void fx_route(');b=s.index('void fx_run(',a)
 s=s[:a]+'''void fx_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){
 mwfx::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());
}
'''+s[b:]
 s=s.replace('std::vector<torch::Tensor> v','std::vector<int64_t> v').replace('std::vector<torch::Tensor>v','std::vector<int64_t>v')
 s=s.replace('int64_t profiling){','int64_t profiling,int64_t M,int64_t cap,int64_t stream_value){')
 s=s.replace('int M=v[11].size(0),cap=v[9].size(0);auto stream=at::cuda::getCurrentCUDAStream();','auto stream=(cudaStream_t)stream_value;')
 s=s.replace('void fx_finish(std::vector<int64_t>v,int64_t start){','void fx_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){').replace('v[11].size(0)','M').replace('at::cuda::getCurrentCUDAStream()','(cudaStream_t)stream')
 s=re.sub(r'v\[(\d+)\]\.data_ptr<([^>]+)>\(\)',lambda m:'('+m[2]+'*)v['+m[1]+']',s)
 s=re.sub(r'v\[(\d+)\]\.data_ptr\(\)',lambda m:'(void*)v['+m[1]+']',s)
 assert 'torch::' not in s and 'at::' not in s
 return PRELUDE+s

def pipeline(s):
 s=lean(s);a=s.index('void pipe_run(');b=s.index(' #define GO',a)
 s=s[:a]+'''void pipe_run(std::vector<int64_t>v,int64_t first,int64_t which,int64_t grid,int64_t M,int64_t cap,int64_t N,int64_t stream_value){
 auto stream=(cudaStream_t)stream_value;
'''+s[b:]
 for i,name in enumerate(('q','s','gs','x','c','off','order','task','slots','y')):
  s=re.sub(r'\b'+name+r'\.data_ptr<([^>]+)>\(\)',lambda m:'('+m[1]+'*)v['+str(i)+']',s)
  s=s.replace(name+'.data_ptr()','(void*)v['+str(i)+']')
 assert 'torch::' not in s and 'at::' not in s
 return PRELUDE+s
