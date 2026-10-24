"""One attention measurement candidate: q1 rescale, q8 denominator, next split kernel."""
import ast
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ATTN=ROOT/'problems/muse_attention_decode'


def assignment(source,name):
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name:
            return ast.literal_eval(n.value)
    raise KeyError(name)


def group_source():
    source=(ATTN/'dev/xqa_group.cu').read_text().replace('#define DEFER_DENOM 0','#define DEFER_DENOM 1')
    wrappers=[]
    for name,op,extra in [('wg8','m64n64k32.f32.e4m3.e4m3',''),('wg16','m64n64k16.f32.f16.f16',',1,0')]:
        registers=','.join('%'+str(i) for i in range(32))
        outputs=','.join(f'"+f"(d[{i//4}][{i%4}])' for i in range(32))
        wrappers.append(f'''DI void {name}(float (&d)[8][4],uint64_t a,uint64_t b){{
#if USE_WGMMA
 asm volatile("{{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.{op} {{{registers}}},%32,%33,p,1,1{extra};}}":{outputs}:"l"(a),"l"(b));
#endif
}}''')
    return source.replace('WG_WRAPPERS','\n'.join(wrappers))


def build():
    base=(ATTN/'cand_v2.py').read_text()
    base=base.replace('def kernel_fn(data: Inputs)','def q1_attention(data: Inputs)')
    c=assignment(base,'CUDA_SRC')
    c=c.replace('#define DEVI __device__ __forceinline__','#define DEVI __device__ __forceinline__\n#ifndef CONDITIONAL_RESCALE\n#define CONDITIONAL_RESCALE 1\n#endif')
    old='''      const float corr = ex2(st.m[mt][rh] - m_use);
      st.m[mt][rh] = m_new;
      st.l[mt][rh] *= corr;
#pragma unroll
      for (int jn = 0; jn < 8; ++jn) {
        st.o[mt][jn][2 * rh] *= corr;
        st.o[mt][jn][2 * rh + 1] *= corr;
      }
'''
    new='''      // The eight query heads have separate maxima. Only skip the block
      // when no row in the entire warp changes: the branch is warp-uniform.
      const bool rescale = !(CONDITIONAL_RESCALE && L == 1) ||
                           __any_sync(0xffffffffu, m_new != st.m[mt][rh]);
      if (rescale) {
        const float corr = ex2(st.m[mt][rh] - m_use);
        st.l[mt][rh] *= corr;
#pragma unroll
        for (int jn = 0; jn < 8; ++jn) {
          st.o[mt][jn][2 * rh] *= corr;
          st.o[mt][jn][2 * rh + 1] *= corr;
        }
      }
      st.m[mt][rh] = m_new;
'''
    assert old in c;c=c.replace(old,new)
    tree=ast.parse(base);node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA_SRC')
    lines=base.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=['CUDA_SRC = '+repr(c)+'\n']
    base=''.join(lines)
    base=base.replace('_EXT = None','Q1_CONDITIONAL = True\n_EXT = {}')
    base=base.replace('if _EXT is None:', 'if Q1_CONDITIONAL not in _EXT:').replace('_EXT = load_inline(', '_EXT[Q1_CONDITIONAL] = load_inline(')
    base=base.replace('name=f"muse_attn_v2c_sm{arch}"','name=f"muse_p5_q1_c{int(Q1_CONDITIONAL)}_sm{arch}"')
    base=base.replace('"--expt-relaxed-constexpr", "-lineinfo"],','"--expt-relaxed-constexpr", "-lineinfo", f"-DCONDITIONAL_RESCALE={int(Q1_CONDITIONAL)}"],')
    base=base.replace('    return _EXT\n','    return _EXT[Q1_CONDITIONAL]\n')
    base=base.replace('key = (q_len, str(device))','key = (q_len, str(device), Q1_CONDITIONAL)')
    prev=(ATTN/'cand_verify8_flashdecode.py').read_text()
    fd=assignment(prev,'VERIFY8_CUDA')
    fd=fd.replace('#define DI __device__ __forceinline__','#define DI __device__ __forceinline__\n#ifndef FD_DEFER_DENOM\n#define FD_DEFER_DENOM 1\n#endif')
    old='  sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);'
    assert old in fd
    fd=fd.replace(old,'  if constexpr(!FD_DEFER_DENOM){\n'+old+'\n  }')
    old='  if(t==0)*(float2*)(out+row*PR+64)=make_float2(st.m[rh],st.l[rh]);'
    new='''  float den=st.l[rh];
  if constexpr(FD_DEFER_DENOM){den+=__shfl_xor_sync(0xffffffff,den,1);den+=__shfl_xor_sync(0xffffffff,den,2);}
  if(t==0)*(float2*)(out+row*PR+64)=make_float2(st.m[rh],den);'''
    assert old in fd;fd=fd.replace(old,new)
    fd=fd.replace('template<int ST,int KT>\n__global__','template<int ST,int KT,bool DEFER=(FD_DEFER_DENOM!=0)>\n__global__')
    driver=(ATTN/'dev/flashdecode/driver.py.inc').read_text().split('def kernel_fn')[0]
    driver=driver.replace('_VERIFY8_EXT = None','Q8_DEFERRED_DENOM = True\n_VERIFY8_EXT = {}')
    driver=driver.replace('if _VERIFY8_EXT is None:', 'if Q8_DEFERRED_DENOM not in _VERIFY8_EXT:')
    driver=driver.replace('_VERIFY8_EXT=load_inline(', '_VERIFY8_EXT[Q8_DEFERRED_DENOM]=load_inline(')
    driver=driver.replace("name=f'muse_flashdecode_q8_sm{arch}'", "name=f'muse_p5_fd_d{int(Q8_DEFERRED_DENOM)}_sm{arch}'")
    driver=driver.replace("'-lineinfo'],verbose=False)", "'-lineinfo',f'-DFD_DEFER_DENOM={int(Q8_DEFERRED_DENOM)}'],verbose=False)")
    driver=driver.replace('return _VERIFY8_EXT\n','return _VERIFY8_EXT[Q8_DEFERRED_DENOM]\n')
    window=(ATTN/'cand_sliding_singlelaunch.py').read_text()
    wdriver=(ATTN/'dev/sliding/driver.py.inc').read_text().split('def kernel_fn')[0]
    group=group_source()
    cpp='''
#include <cstdint>
void xforward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t part,int64_t count,int64_t out,int64_t batch,int64_t hq,int64_t hkv,int64_t stride,int64_t maxpages,int64_t ps,int64_t window,int64_t ns,double scale,int64_t stages,int64_t p8,int64_t stream);
'''
    source=base+'\nVERIFY8_CUDA = '+repr(fd)+'\nVERIFY8_CPP = '+repr(assignment(prev,'VERIFY8_CPP'))+'\n'+driver
    source+='\nWINDOW_CUDA = '+repr(assignment(window,'WINDOW_CUDA'))+'\nWINDOW_CPP = '+repr(assignment(window,'WINDOW_CPP'))+'\n'+wdriver
    source+='\nGROUP_CUDA = '+repr(group)+'\nGROUP_CPP = '+repr(cpp)+'\n'+(HERE/'driver.py.inc').read_text()
    output=ATTN/'cand_p5_rescale_deferred.py';output.write_text(source)
    return output


if __name__=='__main__':
    print(build())
