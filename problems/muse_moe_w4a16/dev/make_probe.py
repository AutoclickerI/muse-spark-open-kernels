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

# Checked V7 control, sharing the prepared bank and synchronization-only scratch.
import ast
v7text=(p/'cand_v7.py').read_text()
v7tree=ast.parse(v7text)
v7lines=v7text.splitlines(keepends=True)
def function_text(name):
 n=next(x for x in v7tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
 return ''.join(v7lines[n.lineno-1:n.end_lineno])
constants={}
for n in v7tree.body:
 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
  for t in n.targets:
   if isinstance(t,ast.Name):constants[t.id]=n.value.value
module=function_text('_module').replace('def _module():','def _v7_probe_module():').replace('_MOD','_V7_MOD').replace('CUDA_SRC','V7_CUDA_SRC').replace('CPP_SRC','V7_CPP_SRC')
call=function_text('kernel_fn').replace('def kernel_fn(data, state):','def _v7_probe_call(data, state):').replace('mod = state["mod"]','mod = state["v7_mod"]')
s+='\nV7_CUDA_SRC = r"""'+constants['CUDA_SRC']+'"""\nV7_CPP_SRC = r"""'+constants['CPP_SRC']+'"""\n_V7_MOD = None\n'+module+'\n\n'+call+'\n'
s=s.replace("    state['grouped_mod'] = _grouped_module()\n", "    state['grouped_mod'] = _grouped_module()\n    state['v7_mod'] = _v7_probe_module()\n")
s=s.replace('n = 9 if skip_native else 12','n = 9 if skip_native else 13')
s=s.replace("    mod.probe_log(ticks, state['probe_count'], m, n, int(skip_native))", '''    if not skip_native:
        checked_output = out
        mod.probe_evict(state['probe_flush'], sink, state['grid'] * 4)
        mod.probe_stamp(ticks, 2 + 2*i)
        _v7_probe_call(data, state)
        mod.probe_stamp(ticks, 3 + 2*i)
        out = checked_output
    mod.probe_log(ticks, state['probe_count'], m, n, int(skip_native))''')
(p/'cand_probe.py').write_text(s)

# Canonical K512 bulk reader, still over the same resident bank.
tma_tree=ast.parse((p/'cand_rs_tma.py').read_text())
tma_node=next(n for n in tma_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GROUPED_SRC' for t in n.targets))
tma_src=ast.literal_eval(tma_node.value)
a=tma_src.index('namespace canonical {');b=tma_src.index('void route(torch::Tensor');c=tma_src.index('void configure()')
tma_code=(tma_src[a:b]+tma_src[c:]).replace('namespace canonical {','namespace canonical_tma {').replace('canonical::','canonical_tma::').replace('void configure()','void configure_tma()').replace('void gemm(torch::Tensor','void gemm_tma(torch::Tensor')
s+='\nGROUPED_SRC += r"""'+tma_code+'"""\n'
s=s.replace("'gemm_wide']", "'gemm_wide', 'gemm_tma', 'configure_tma', 'probe_compare']")
s=s.replace("'void probe_evict(torch::Tensor,torch::Tensor,int64_t);", "'void gemm_tma(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);\\n'\n                            'void configure_tma();\\n'\n                            'void probe_compare(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\\n'\n                            'void probe_evict(torch::Tensor,torch::Tensor,int64_t);")
s=s.replace('void probe_log(torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);','void probe_log(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);')
s=s.replace("    state['v7_mod'] = _v7_probe_module()\n", "    state['v7_mod'] = _v7_probe_module()\n    state['grouped_mod'].configure_tma()\n    state['weight_stages'] = 2 if torch.cuda.get_device_capability() == (9, 0) else 1\n")
s=s.replace('def _grouped_call(data, state, fixed_n=0, stages=2, wide=False):','def _grouped_call(data, state, fixed_n=0, stages=2, wide=False, tma=False):')
s=s.replace('    runner = mod.gemm_wide if wide else mod.gemm', "    runner = mod.gemm_tma if tma else (mod.gemm_wide if wide else mod.gemm)\n    extra = (state['weight_stages'],) if tma else ()")
s=s.replace('fixed_n, grid, stages)', 'fixed_n, grid, stages, *extra)')

# Emit independent peer-error diagnostics for every timed result, without changing output checks.
s=s.replace('const int64_t* ticks,int* count,int m,int n,int skip_native', 'const int64_t* ticks,int* count,const float* stats,int m,int n,int skip_native')
s=s.replace('int v=skip_native?i+3:i;', '''int v=skip_native?(i<9?i+3:13):i;
  float err=0.f,den=0.f,bad=0.f,worst=0.f;
  for(int r=0;r<m;++r){const float* q=stats+(i*m+r)*3;err+=q[0];den+=q[1];bad+=q[2];}
  float floor=den/m*.01f;
  for(int r=0;r<m;++r){const float* q=stats+(i*m+r)*3;worst=fmaxf(worst,q[0]/fmaxf(q[1],fmaxf(floor,1e-30f)));}''')
s=s.replace('variant=%d raw_ns=%lld\\n",m,v,(long long)(ticks[3+2*i]-ticks[2+2*i])', 'variant=%d raw_ns=%lld peer_l2=%.7f peer_max_row=%.7f finite=%d\\n",m,v,(long long)(ticks[3+2*i]-ticks[2+2*i]),sqrtf(err/fmaxf(den,1e-30f)),sqrtf(worst),(int)(bad==0.f)')
s=s.replace('void probe_log(torch::Tensor ticks,torch::Tensor count,int64_t m,int64_t n,int64_t skip_native)', 'void probe_log(torch::Tensor ticks,torch::Tensor count,torch::Tensor stats,int64_t m,int64_t n,int64_t skip_native)')
s=s.replace('ticks.data_ptr<int64_t>(),count.data_ptr<int>(),m,n,skip_native', 'ticks.data_ptr<int64_t>(),count.data_ptr<int>(),stats.data_ptr<float>(),m,n,skip_native')
s+=r'''
GROUPED_SRC += r"""
namespace probe {
__global__ void compare(const __nv_bfloat16* actual,const __nv_bfloat16* ref,float* stats,int variant,int m){
 __shared__ float partial[8][3];
 int row=blockIdx.x,t=threadIdx.x,l=t&31,w=t>>5;
 float err=0.f,den=0.f,bad=0.f;
 for(int c=t;c<4096;c+=256){float a=__bfloat162float(actual[(size_t)row*4096+c]),b=__bfloat162float(ref[(size_t)row*4096+c]);float d=a-b;err+=d*d;den+=b*b;bad+=(!isfinite(a)||!isfinite(b));}
 for(int d=16;d;d>>=1){err+=__shfl_down_sync(~0u,err,d);den+=__shfl_down_sync(~0u,den,d);bad+=__shfl_down_sync(~0u,bad,d);}
 if(l==0){partial[w][0]=err;partial[w][1]=den;partial[w][2]=bad;}
 __syncthreads();
 if(w==0){err=l<8?partial[l][0]:0;den=l<8?partial[l][1]:0;bad=l<8?partial[l][2]:0;
  for(int d=16;d;d>>=1){err+=__shfl_down_sync(~0u,err,d);den+=__shfl_down_sync(~0u,den,d);bad+=__shfl_down_sync(~0u,bad,d);}
  if(l==0){float* q=stats+(variant*m+row)*3;q[0]=err;q[1]=den;q[2]=bad;}
 }
}
}
void probe_compare(torch::Tensor actual,torch::Tensor ref,torch::Tensor stats,int64_t variant,int64_t m){probe::compare<<<m,256,0,at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)actual.data_ptr(),(const __nv_bfloat16*)ref.data_ptr(),stats.data_ptr<float>(),variant,m);C10_CUDA_KERNEL_LAUNCH_CHECK();}
"""
'''
# Rewrite the Python sweep plainly; the returned result remains the 128-row three-stage RS path.
tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn')
lines=s.splitlines(keepends=True);a=sum(map(len,lines[:fn.lineno-1]));b=sum(map(len,lines[:fn.end_lineno]))
fntext='''def kernel_fn(data, state):
    # Check-only diagnostic: all work, timing, flushing and validation occurs in this call.
    m = data['hidden_states'].shape[0]
    assert m <= 16384
    skip_native = m > 512
    specs = [] if skip_native else [('native', 2, 2), ('native', 2, 1), ('native', 3, 1)]
    specs += [('grouped', width, 2) for width in (8, 16, 32, 64, 128, 0)]
    specs += [('grouped', 0, 3), ('wide', 0, 2), ('wide', 0, 3)]
    if not skip_native:
        specs += [('v7', 0, 0)]
    specs += [('tma', 0, 2)]
    n = len(specs)
    dev = data['hidden_states'].device
    ticks = torch.empty((2*n+2,), dtype=torch.int64, device=dev)
    stats = torch.empty((n,m,3), dtype=torch.float32, device=dev)
    sink = torch.empty((state['grid']*4*256,), dtype=torch.uint32, device=dev)
    mod = state['grouped_mod']
    mod.probe_stamp(ticks, 0)
    mod.probe_stamp(ticks, 1)
    anchor = None
    checked_output = None
    for i, (kind, a, b) in enumerate(specs):
        mod.probe_evict(state['probe_flush'], sink, state['grid']*4)
        mod.probe_stamp(ticks, 2+2*i)
        if kind == 'native':
            out = _native_call(data, state, a, b)
        elif kind == 'v7':
            out = _v7_probe_call(data, state)
        else:
            out = _grouped_call(data, state, a, b, wide=(kind=='wide'), tma=(kind=='tma'))
        mod.probe_stamp(ticks, 3+2*i)
        if anchor is None:
            anchor = out
        mod.probe_compare(out, anchor, stats, i, m)
        if kind == 'wide' and b == 3:
            checked_output = out
    mod.probe_log(ticks, state['probe_count'], stats, m, n, int(skip_native))
    return checked_output
'''
s=s[:a]+fntext+s[b:]
(p/'cand_probe.py').write_text(s)
