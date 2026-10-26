"""Per-segment finite B-fill SASS model,unchanged row/pass controls,not measured stalls."""
import collections,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_intpipe'
def fill(cu):a=cu.index('#pragma unroll\n      for (int j = 0; j < NT; ++j) {',cu.index('auto unit_body'));b=cu.index('    }\n    if constexpr (TRACE)',a);return cu[a:b]
base=(O/'base1.cu').read_text();new=(O/'affine1.cu').read_text();a=new.index('template<bool CG,int OFF>');b=new.index('// 16 E2M1 codes',a);helper=new[a:b]
header='''#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
__device__ __forceinline__ unsigned long long mark(){unsigned long long t;asm volatile("mov.u64 %0,%%clock64;":"=l"(t)::"memory");return t;}
'''+helper+'''
template<int MODE,int PASS> __global__ void ip_fill(const __nv_bfloat16* x,const __nv_bfloat16* inter,const int16_t* pairs,uint32_t* output,unsigned long long* stamps,int ne,int off,int nb){
 __shared__ int16_t pairbuf[256];int tid=threadIdx.x,lane=tid&31,warp=tid>>5,q=lane&3;constexpr int NT=1,NB=8,HD=4096;int pass=PASS;
 pairbuf[tid]=pairs[tid];__syncthreads();const int16_t* pair_of=pairbuf;uint32_t Bf[1][8][8];auto t0=mark();
 if constexpr(MODE==0){
'''
tail='''
 }
 auto t1=mark();
#pragma unroll
 for(int g=0;g<8;++g){
#pragma unroll
 for(int k=0;k<8;++k)output[tid*64+g*8+k]=Bf[0][g][k];}
 stamps[tid*2]=t0;stamps[tid*2+1]=t1;}
'''
s=header+fill(base)+'\n}else{\n'+fill(new)+tail
for mode in (0,1):
 for pas in (0,1):s+=f'\ntemplate __global__ void ip_fill<{mode},{pas}>(const __nv_bfloat16*,const __nv_bfloat16*,const int16_t*,uint32_t*,unsigned long long*,int,int,int);'
p=O/'marked_fill.cu';p.write_text(s);proc=subprocess.run(['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-c',str(p),'-o',str(p.with_suffix('.o')),'-Xptxas=-v'],capture_output=True,text=True,check=True);p.with_suffix('.compile.log').write_text(proc.stderr);sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass);out=[]
for name,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
 ins=[m[1] for m in re.finditer(r'/\*[0-9a-f]+\*/\s+(.*?);',body)];marks=[i for i,v in enumerate(ins) if 'CS2R' in v and 'SR_CLOCK' in v];assert len(marks)==2;part=ins[marks[0]+1:marks[1]];ops=collections.Counter(re.sub(r'^@!?(?:U?P)\d+\s+','',v).split()[0].split('.')[0] for v in part)
 kinds=('SHF','LOP3','LEA','IMAD','IADD3','VIADD','ISETP','SEL','PLOP3');count=sum(ops[k] for k in kinds);mode,pas=map(int,re.search(r'ip_fillILi(\d)ELi(\d)E',name).groups());out.append(dict(arm='BASE' if mode==0 else 'AFFINE',pass_=pas,INT=count,LDG=ops['LDG'],ops=dict(ops)))
assert len(out)==4 and all(v['LDG']==16 for v in out)
r=dict(rows=out,normalization='Per warp per expert/token-block segment;divide INT by256*L weights to normalize per requested weight,then by1.96 empirical issue rate for cycles. L is that segment tile count. Does not include decoder,copy schedule or epilogue.',source='Exact extracted S19 BASE/AFFINE fill bodies;marked probe changes register/control context,so full-kernel totals remain separately recorded.')
r['INT_fill_cycles_per_requested_weight_times_segment_L']={v['arm']+'_'+str(v['pass_']):v['INT']/(256*1.96) for v in out}
(D/'fill_model.json').write_text(json.dumps(r,indent=2)+'\n');print([(v['arm'],v['pass_'],v['INT'],v['LDG']) for v in out])
