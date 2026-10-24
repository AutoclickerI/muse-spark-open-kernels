"""Expert-wide M128 MoE: verified distributed finalizer, then shared A reuse."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent;M128=HERE.parent;ROOT=HERE.parents[5]
sys.path[:0]=[str(M128),str(M128.parent),str(ROOT/'tools')]
from source import source
from build_repair_lab import pointer_bindings,rename
from native_abi import PRELUDE,CPP_PRELUDE
from build_prefill_lab import subset
from okc_run import rejected_words
spec=importlib.util.spec_from_file_location('scale_builder',M128/'scale_bank/build.py')
scale=importlib.util.module_from_spec(spec);spec.loader.exec_module(scale)


def finalizer(s,prefix):
    begin=s.index('   __threadfence();__syncthreads();\n   if(t==0)last=(publish(group_ready+group)==63);')
    end=s.index('\n  }\n  __syncthreads();\n }\n}',begin)
    s=s[:begin]+s[end:]
    s=s.replace('__shared__ int last,flags[NT];','__shared__ int last;')
    ref=(M128.parent/'w2fix.cu').read_text();ref=ref[ref.index('namespace mwfx {\n__global__ void finish'):]
    ref=ref.replace('mwfx',prefix).replace('fx_finish',prefix+'_finish')
    return s+'\n'+ref


def rs_source():
    rec=scale.recode_source(source())
    s=pointer_bindings(rename(rec,'rot64',(3,4,5))).replace('rot64','or_rs')
    return finalizer(s,'or_rs')


def mma_ss():
    body='template<int NW>__device__ __forceinline__ void ss_mma(float*c,const __nv_bfloat16*as,const __nv_bfloat16*bs,int atom){\n'
    body+='\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n'
    body+=' uint64_t da=(uint64_t)((smaddr(as)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n'
    body+=' uint64_t db=(uint64_t)((smaddr(bs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);da+=atom*2;db+=atom*2;\n'
    for i,n in enumerate((16,32,64,128)):
        body+=(' if' if i==0 else ' else if')+f' constexpr(NW=={n}){{\n'
        regs=','.join('%'+str(j) for j in range(n//2));outs=','.join('"+f"(c['+str(j)+'])' for j in range(n//2))
        body+=f' asm volatile("{{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 '
        body+='{' +regs+'},%'+str(n//2)+',%'+str(n//2+1)+',p,1,1,0,0;}"\n :'+outs+'\n :"l"(da),"l"(db));\n }\n'
    body+='''#else
 int t=threadIdx.x,w=t>>5,l=t&31;unsigned a[4];
 #pragma unroll
 for(int j=0;j<4;++j){int r=w*16+(l>>2)+(j&1)*8,k=atom*16+(l&3)*2+(j>>1)*8;
  a[j]=*(const unsigned*)(as+r*64+(k^((r&7)*8)));}
 mma<NW>(c,a,bs,atom);
#endif
}
'''
    return body


def ss_source(limit):
    s=scale.recode_source(source())
    s=s.replace('namespace mw128','namespace or_ss'+str(limit)).replace('mw128::','or_ss'+str(limit)+'::')
    prefix='or_ss'+str(limit)
    s=s.replace('m128_',prefix+'_')
    a=s.index('template<int NT,int WIDTH,int KT,int S,int SPLIT>')
    b=s.index('template<int NT,int KT,int S,int SPLIT,bool ADAPT,bool FIRST>',a)
    core='''template<int NT,int WIDTH,int KT,int S,int SPLIT>
__device__ __forceinline__ void core(const int*Q,const unsigned char*SCALE,const __nv_bfloat16*X,
 const int*src,int e,int mt,int N,int split,unsigned char*qs,unsigned char*ss,
 __nv_bfloat16*bs,__nv_bfloat16*as,float*acc){
 static_assert(KT==64&&S==1&&SPLIT==1);
 int t=threadIdx.x,w=t>>5,l=t&31;
 #pragma unroll 1
 for(int kstep=0;kstep<64;++kstep){
  load_stage<NT,64>(qs,ss,bs,Q,SCALE,X,src,e,mt,N,kstep,WIDTH);
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
  unsigned af[16];
  #pragma unroll
  for(int g=0;g<2;++g){uint2 q=*(const uint2*)(qs+(g*128+t)*8);
   unsigned sf=*(const unsigned*)(ss+g*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af+g*8);}
  // One dequantization and one shared A tile for every N subgroup. The
  // register ownership is the actual RS m64k16 layout used by the incumbent.
  #pragma unroll
  for(int atom=0;atom<4;++atom){
   #pragma unroll
   for(int j=0;j<4;++j){int r=w*16+(l>>2)+(j&1)*8,c=atom*16+(l&3)*2+(j>>1)*8;
    *(unsigned*)(as+r*64+(c^((r&7)*8)))=af[atom*4+j];}
  }
  __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
  #pragma unroll
  for(int tile=0;tile<(WIDTH+NLIMIT-1)/NLIMIT;++tile){
   constexpr int NW=WIDTH<NLIMIT?WIDTH:NLIMIT;
   #pragma unroll
   for(int atom=0;atom<4;++atom)ss_mma<NW>(acc+tile*(NLIMIT/2),as,bs+tile*NLIMIT*64,atom);
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;":::"memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
  __syncthreads();
 }
}
'''.replace('NLIMIT',str(limit))
    s=s[:a]+mma_ss()+core+s[b:]
    s=s.replace('auto*bs=(__nv_bfloat16*)(sm+BS);','auto*bs=(__nv_bfloat16*)(sm+BS);auto*as=bs+S*NT*KT;')
    s=s.replace('qs,ss,bs,acc);','qs,ss,bs,as,acc);')
    s=s.replace('  if constexpr(ADAPT){','  if constexpr(NT>128)core<NT,NT,KT,S,SPLIT>(Q,SCALE,X,src,e,mt,N,split,qs,ss,bs,as,acc);\n  else if constexpr(ADAPT){',1)
    s=s.replace('FIRST?64*NT:1','FIRST?32*NT:1')
    old='''   if(t<NT&&row+t<count){float sum=0;
    #pragma unroll
    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];
    SS[(size_t)(offset+row+t)*64+mt]=sum;
   }'''
    new='''   for(int rr=t;rr<NT;rr+=128)if(row+rr<count){float sum=0;
    #pragma unroll
    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+rr];
    SS[(size_t)(offset+row+rr)*64+mt]=sum;
   }'''
    assert old in s
    s=s.replace(old,new)
    # Only up rows need shared storage; gate rows stay in their FP32 registers
    # until they are rounded to BF16 exactly as in the original epilogue.
    begin=s.index('  if constexpr(FIRST){',s.index('__global__ void __launch_bounds__'))
    end=s.index('  }else{\n   float gs=GS[e]*256.f;',begin)
    epilogue='''  if constexpr(FIRST){
   #pragma unroll
   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,c=w*16+(l>>2)+((j>>1)&1)*8;
    if(c>=32)epi[(c-32)*NT+r]=__float2bfloat16_rn(acc[j]*GS[e+32]*256.f);
   }
   __syncthreads();
   #pragma unroll
   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,c=w*16+(l>>2)+((j>>1)&1)*8;
    if(c<32&&row+r<count){float g=__bfloat162float(__float2bfloat16_rn(acc[j]*GS[e]*256.f));
     float u=__bfloat162float(epi[c*NT+r]);
     ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);}
   }
'''
    s=s[:begin]+epilogue+s[end:]
    a=s.index('#define FT_CASES(F)');b=s.index('\n}',a)
    s=s[:a]+'#define FT_CASES(F) \\\n F(0,128,64,1,1,true) \\\n F(1,256,64,1,1,false) \\\n F(2,384,64,1,1,false)'+s[b:]
    s=s.replace('+S*NT*KT*2;','+S*NT*KT*2+8192;')
    # Reuse the pointer conversion helper's explicit wrapper marker.
    s=s.replace(prefix,'rot64');s=pointer_bindings(s).replace('rot64',prefix)
    return finalizer(s,prefix)


def main():
    codes={'or_rs':rs_source(),'or_ss64':ss_source(64),'or_ss128':ss_source(128)}
    prior=ROOT/'prof/lab/moe_m128_issue_limit.py'
    old=ast.parse(prior.read_text())
    def value(name):return next(ast.literal_eval(n.value) for n in old.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets))
    # Use the byte-identical actual recoded NT1 native control from the prior lab.
    nt=(M128/'issue_limit/norig.cu').read_text();codes['norig']=nt
    cpp=CPP_PRELUDE;funs=[]
    for p in ('or_rs','or_ss64','or_ss128'):
        cpp+=f'int64_t {p}_init();std::vector<int64_t> {p}_info(int64_t,int64_t);void {p}_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void {p}_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void {p}_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);\n'
        funs += [p+'_'+f for f in ('init','info','route','run','finish')]
    cpp+='int64_t norig_init();int64_t norig_smem(int64_t,int64_t);int64_t norig_resident(int64_t,int64_t);void norig_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);'
    funs += ['norig_'+f for f in ('init','smem','resident','run')]
    helpers=subset(prior.read_text(),('dc_weights','dc_fixtures','dc_oracles','MWGraph','mw_repack_planes','_encode_scales','issue_prepare','issue_nt_prepare'))
    tree=ast.parse(helpers)
    # The only dependency on the old native module is replaced by this lab's
    # module. Remove the unused old build and embedded CUDA strings.
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='issue_module':n.func.id='or_module'
    tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name in ('issue_module','_v9_module'))
        and not (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('ISSUE_CUDA','ISSUE_CPP','ISSUE_FUNCTIONS','ISSUE_OPTS') for t in n.targets))]
    for p,s in codes.items():(HERE/(p+'.cu')).write_text(s)
    cuda='\n'.join(codes.values());text=ast.unparse(tree)+'\nimport functools\nimport gc\nfrom torch.utils.cpp_extension import load_inline\n'
    text+='OR_CUDA='+repr(cuda)+'\nOR_CPP='+repr(cpp)+'\nOR_FUNCTIONS='+repr(funs)+'\n'+(HERE/'driver.py.inc').read_text()
    assert not rejected_words(text),rejected_words(text)
    out=ROOT/'prof/lab/moe_m128_no_reread.py';out.write_text(text)
    manifest=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),
        sources={p:hashlib.sha256(s.encode()).hexdigest() for p,s in codes.items()},
        prior_scale_receipt='471f6d6216804dda',finalizer_receipt='97be11d4631b44d6',issue_receipt='dc3fea0a20a349bf',
        token_capacities=[128,256,384],shared_N_limits=[64,128],bank='same-size uint8 exact recode',
        guarantee='One expert token group per touched expert at these capacities; each K64 packed tile loaded/dequantized once then applied to every N subgroup before reuse.')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out.relative_to(ROOT),manifest['sha256'],len(text))


if __name__=='__main__':main()
