"""Generate canonical BN64/128/256 register-A WGMMA and timed DG hybrid."""
from pathlib import Path
import re
P=Path(__file__).resolve().parents[1]
s=(P/'dev/canonical_grouped_rs128_separate.cu').read_text()
a=s.index('   if constexpr (N == 8)');b=s.index('\n }\n#else',a)
atoms=[]
for n in (8,16,32,64,128,256):
 regs=n//2
 outs=', '.join('%'+str(i) for i in range(regs))
 ins=', '.join('%'+str(regs+i) for i in range(4))
 constraints=', '.join(f'"+f"(d[{i}])' for i in range(regs))
 atoms.append(f'''  {'if' if n==8 else 'else if'} constexpr (N == {n}) {{
    asm volatile("{{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{outs}}}, {{{ins}}}, %{regs+4}, p, 1, 1, 0; }}\\n"
      : {constraints}
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc));
  }}''')
s=s[:a]+'\n'.join(atoms)+s[b:]
s=s.replace('template<int N,int MODE,int STAGES>','template<int N,int MODE,int STAGES,int RB>')
s=s.replace('MTILES=MODE?512:256, RB=2,','MTILES=MODE?512:256,')
a=s.index('void gemm(torch::Tensor');s=s[:a]+'''void gemm(torch::Tensor w,torch::Tensor gs,torch::Tensor gs_up,torch::Tensor x,torch::Tensor out,
 torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t width,int64_t grid,int64_t stages,int64_t rb){
 auto stream=at::cuda::getCurrentCUDAStream();
#define RUN(N,M,S,R) do { \\
 cudaFuncSetAttribute(canonical::grouped<N,M,S,R>,cudaFuncAttributeMaxDynamicSharedMemorySize,N*128*S); \\
 canonical::grouped<N,M,S,R><<<grid,128,N*128*S,stream>>>(w.data_ptr<uint8_t>(),gs.data_ptr<float>(),gs_up.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33); } while(0)
#define PICK(M) if(rb==1){if(width==64){RUN(64,M,2,1);}else if(width==128){RUN(128,M,2,1);}else if(stages==2){RUN(256,M,2,1);}else{RUN(256,M,3,1);}} \\
 else if(width==0){RUN(8,M,2,2);RUN(16,M,2,2);RUN(32,M,2,2);RUN(64,M,2,2);RUN(128,M,2,2);} \\
 else if(width==64){RUN(64,M,2,2);}else if(stages==2){RUN(128,M,2,2);}else{RUN(128,M,3,2);}
 if(mode){PICK(1);}else{PICK(0);}
#undef PICK
#undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
s+='\n'+(P/'dev/moe_w4a16_canonical.cu').read_text()
(P/'dev/canonical_portfolio.cu').write_text(s)
print('wrote canonical_portfolio.cu')
import ast
base=(P/'cand_canonical_grouped.py').read_text()
front=base[:base.index('\nGROUPED_SRC =')]
front=front.replace('moe = experts_grouped(', 'moe = _experts_selected(')
front=front.replace('Prefill with one canonical V6 expert bank and adaptive RS128 WGMMA groups.',
    'Prefill expert portfolio: canonical NVFP4 bank, BN64/128/256 and timed BF16 scratch.')
def definition(name):
 n=next(n for n in ast.parse(base).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 start=min([n.lineno]+[d.lineno for d in n.decorator_list]);return '\n'.join(base.splitlines()[start-1:n.end_lineno])+'\n\n'
meta=definition('_canonical_pairs')+definition('_canonical_finalize')
pack=base[base.index('_DQ_SLOTS ='):base.index('\ndef prepare(weights):',base.index('_DQ_SLOTS ='))]
dqsrc=(P/'dev/canonical_dequant.py').read_text()
dqtree=ast.parse(dqsrc)
dq=''
for node in dqtree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ('_bf16x2_mul','_canonical_expand_coalesced'):
        first=min([node.lineno]+[d.lineno for d in node.decorator_list])
        dq+='\n'.join(dqsrc.splitlines()[first-1:node.end_lineno])+'\n\n'
host=(P/'dev/portfolio_host.py').read_text()
pipe=(P/'dev/moe_canonical_host.py').read_text()
i=pipe.index('@triton.jit\ndef tile_offsets_kernel')
j=pipe.index('_DQ_SLOTS =',i)
pipe=pipe[i:j]
# This wrapper is a measured contender on the same bank, not another resident layout.
pipe=pipe.replace('def experts_tiled(', 'def _experts_pipeline(')
pipe='\nBN=128\nD=4096\n'+pipe
host=pipe+'\n'+host
prepare='''

def prepare(weights):
    st=prepare_base(weights)
    st['w13p']=repack_v6(weights['w13'],weights['w13_sf'],True)
    st['w2p']=repack_v6(weights['w2'],weights['w2_sf'],False)
    st['gs13_gate']=weights.get('gs13_gate',weights['w13_gs']).float().contiguous()
    st['gs13_up']=weights.get('gs13_up',weights['w13_gs']).float().contiguous()
    st['gs2']=weights['w2_gs'].float().contiguous()
    st['post_expert']=st['post_expert_bf16']
    st['grouped_mod']=_grouped_module()
    st['tiled_mod']=st['grouped_mod']
    st['sms']=torch.cuda.get_device_properties(weights['qkv_gate'].device).multi_processor_count
    for key in ('w13','w13_sf','w2','w2_sf'):
        st.pop(key,None)
    st['expert_dispatch']={753:'adaptive128',3012:'bn128_r2',16384:'dg_hint'}
    if IS_HOPPER:
        _tune_portfolio(st)
    return st
'''
out=front+'\nGROUPED_SRC = '+repr(s)+'\n'+meta+pack+dq+'\n'+host+'\n'+prepare
(P/'cand_expert_portfolio.py').write_text(out)
print('wrote cand_expert_portfolio.py')
