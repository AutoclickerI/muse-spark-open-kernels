from pathlib import Path
p=Path(__file__).resolve().parent.parent
s=(p/'cand_rs.py').read_text()
s=s.replace('def _native_call(data, state):','def _native_call(data, state, requested_stages=2, grid_cap=2):')
s=s.replace('    nstage = 2\n','    nstage = requested_stages\n',1)
s=s.replace('state["grid"] * mod.resident_blocks(nt, nstage, M)',
            'state["grid"] * min(grid_cap, mod.resident_blocks(nt, nstage, M))')
# Add helpers to this diagnostic module only. The production source has no timers/logging.
start=s.index('\nGROUPED_SRC = ')
end=s.index('\nimport os',start)
s=s[:end]+'''\nGROUPED_SRC += r"""
namespace probe {
__global__ void evict(const uint32_t* data,uint32_t* sink,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x,step=gridDim.x*blockDim.x;
 uint32_t sum=0;
 for(int j=i;j<n;j+=step){uint32_t v;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(v):"l"(data+j):"memory");sum^=v;}
 sink[i]=sum;
}
__global__ void stamp(int64_t* values,int i){
 uint64_t now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));values[i]=now;
}
__global__ void log(const int64_t* ticks,int* count,int m,int n,int skip_native){
 unsigned call=atomicAdd(count+m,1)+1; if(call&(call-1))return;
 printf("MUSE_SWEEP m=%d call=%u stamp_control_ns=%lld\\n",m,call,(long long)(ticks[1]-ticks[0]));
 for(int i=0;i<n;++i){
  int v=skip_native?i+3:i;
  printf("MUSE_SWEEP m=%d variant=%d raw_ns=%lld\\n",m,v,(long long)(ticks[3+2*i]-ticks[2+2*i]));
 }
}
}
void probe_evict(torch::Tensor data,torch::Tensor sink,int64_t grid){probe::evict<<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(data.data_ptr<uint32_t>(),sink.data_ptr<uint32_t>(),data.numel());C10_CUDA_KERNEL_LAUNCH_CHECK();}
void probe_stamp(torch::Tensor ticks,int64_t i){probe::stamp<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(ticks.data_ptr<int64_t>(),i);C10_CUDA_KERNEL_LAUNCH_CHECK();}
void probe_log(torch::Tensor ticks,torch::Tensor count,int64_t m,int64_t n,int64_t skip_native){probe::log<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(ticks.data_ptr<int64_t>(),count.data_ptr<int>(),m,n,skip_native);C10_CUDA_KERNEL_LAUNCH_CHECK();}
"""
''' +s[end:]
s=s.replace("cuda_sources=GROUPED_SRC, functions=['route', 'gemm'],",
'''cuda_sources=GROUPED_SRC, functions=['route', 'gemm', 'probe_stamp', 'probe_log', 'probe_evict'],''')
s=s.replace("'void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);',",
"""'void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);\\n'
                            'void probe_evict(torch::Tensor,torch::Tensor,int64_t);\\n'
                            'void probe_stamp(torch::Tensor,int64_t);\\n'
                            'void probe_log(torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);',""")
s=s.replace("    state['grouped_mod'] = _grouped_module()\n", "    state['grouped_mod'] = _grouped_module()\n    state['probe_count'] = torch.zeros((16385,), dtype=torch.int32, device=state['w13p'].device)\n    state['probe_flush'] = torch.zeros((256 * 1024 * 1024 // 4,), dtype=torch.uint32, device=state['w13p'].device)\n")
pos=s.index('\ndef kernel_fn(data, state):',s.index('def _grouped_call'))
s=s[:pos]+'''
def kernel_fn(data, state):
    # CHECK-ONLY diagnostic. Every variant runs on every fresh call; no autotuning,
    # output caching, input-dependent host choices or changes to judge timing.
    m = data['hidden_states'].shape[0]
    assert m <= 16384
    skip_native = m > 512
    n = 7 if skip_native else 10
    ticks = torch.empty((2*n + 2,), dtype=torch.int64, device=data['hidden_states'].device)
    mod = state['grouped_mod']
    mod.probe_stamp(ticks, 0)
    mod.probe_stamp(ticks, 1)
    sink = torch.empty((state['grid'] * 4 * 256,), dtype=torch.uint32, device=data['hidden_states'].device)
    i = 0
    if not skip_native:
        for stages, cap in ((2, 2), (2, 1), (3, 1)):
            mod.probe_evict(state['probe_flush'], sink, state['grid'] * 4)
            mod.probe_stamp(ticks, 2 + 2*i)
            out = _native_call(data, state, stages, cap)
            mod.probe_stamp(ticks, 3 + 2*i)
            i += 1
    for width in (8, 16, 32, 64, 128, 0):
        mod.probe_evict(state['probe_flush'], sink, state['grid'] * 4)
        mod.probe_stamp(ticks, 2 + 2*i)
        out = _grouped_call(data, state, width, 2)
        mod.probe_stamp(ticks, 3 + 2*i)
        i += 1
    mod.probe_evict(state['probe_flush'], sink, state['grid'] * 4)
    mod.probe_stamp(ticks, 2 + 2*i)
    out = _grouped_call(data, state, 0, 3)
    mod.probe_stamp(ticks, 3 + 2*i)
    mod.probe_log(ticks, state['probe_count'], m, n, int(skip_native))
    return out
'''
s=s.replace('"""NVFP4 MoE:', '"""Diagnostic sweep over a single canonical NVFP4 bank. NVFP4 MoE:',1)
(p/'cand_probe.py').write_text(s)

# Two-fragment WGMMA control: same single prepared weight bank and shared routing API.
import ast
wide_tree=ast.parse((p/'cand_rs128.py').read_text())
wide_node=next(n for n in wide_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GROUPED_SRC' for t in n.targets))
wide_src=ast.literal_eval(wide_node.value)
a=wide_src.index('namespace canonical {')
b=wide_src.index('void route(torch::Tensor')
c=wide_src.index('void gemm(torch::Tensor')
wide_code=(wide_src[a:b]+wide_src[c:]).replace('namespace canonical {','namespace canonical_wide {').replace('canonical::','canonical_wide::').replace('void gemm(torch::Tensor','void gemm_wide(torch::Tensor')
s+='\nGROUPED_SRC += r"""'+wide_code+'"""\n'
s=s.replace("functions=['route', 'gemm', 'probe_stamp', 'probe_log', 'probe_evict']", "functions=['route', 'gemm', 'probe_stamp', 'probe_log', 'probe_evict', 'gemm_wide']")
s=s.replace("'void probe_evict(torch::Tensor,torch::Tensor,int64_t);", "'void gemm_wide(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);\\n'\n                            'void probe_evict(torch::Tensor,torch::Tensor,int64_t);")
s=s.replace('def _grouped_call(data, state, fixed_n=0, stages=2):','def _grouped_call(data, state, fixed_n=0, stages=2, wide=False):')
s=s.replace("    grid = state['grid'] * 4\n    mod.gemm(", "    grid = state['grid'] * 4\n    runner = mod.gemm_wide if wide else mod.gemm\n    runner(")
s=s.replace("    mod.gemm(state['w2p']", "    runner(state['w2p']")
s=s.replace('n = 7 if skip_native else 10','n = 9 if skip_native else 12')
s=s.replace('    mod.probe_log(ticks, state[\'probe_count\'], m, n, int(skip_native))', '''    i += 1
    for stages in (2, 3):
        mod.probe_evict(state['probe_flush'], sink, state['grid'] * 4)
        mod.probe_stamp(ticks, 2 + 2*i)
        out = _grouped_call(data, state, 0, stages, wide=True)
        mod.probe_stamp(ticks, 3 + 2*i)
        i += 1
    mod.probe_log(ticks, state['probe_count'], m, n, int(skip_native))''')
(p/'cand_probe.py').write_text(s)
