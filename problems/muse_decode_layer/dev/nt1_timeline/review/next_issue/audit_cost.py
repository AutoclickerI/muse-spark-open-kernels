"""Count actual sm90a bodies and marked B-fill paths; static counts are not stall counters."""
import collections,hashlib,itertools,json,re,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_next_issue';sys.path.insert(0,str(D));import build
INT=('SHF','LOP3','LEA','IMAD','IADD3','VIADD','ISETP','PLOP3')
def body(s,needle):return next(b for n,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',s,re.S) if needle in n)
def instructions(b):
 return [(int(m[1],16),m[2]) for line in b.splitlines() if (m:=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;',line))]
def ops(ins):return dict(collections.Counter(re.sub(r'^@!?P\d+\s+','',o).split()[0].split('.')[0] for pc,o in ins))
def compile(cu):
 obj=cu.with_suffix('.o');r=subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_90a','-lineinfo','-c',str(cu),'-o',str(obj),'-Xptxas=-v'],capture_output=True,text=True);cu.with_suffix('.compile.log').write_text(r.stdout+r.stderr);r.check_returncode();s=subprocess.check_output(['cuobjdump','--dump-sass',str(obj)],text=True);cu.with_suffix('.sass').write_text(s);return s,r.stderr
def full(cu):
 s,log=compile(cu);out={}
 for flag in (69,95):
  b=body(s,f'moe_varILi1ELi{flag}E');ins=instructions(b);op=ops(ins)
  resource=re.search(r'Function properties for (\S*moe_varILi1ELi'+str(flag)+r'E\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers',log)
  assert resource and op['HMMA']==32 and op['HMUL2']==128
  out[str(flag)]=dict(instructions=len(ins),text_bytes=max(pc for pc,o in ins)+16,ops=op,integer_address_control=sum(op.get(k,0) for k in INT),registers=int(resource[5]),stack_bytes=int(resource[2]),spill_store_bytes=int(resource[3]),spill_load_bytes=int(resource[4]))
 return out
def main():
 s15=build.review.original()[2];cancel=build.review.source('CANCEL',rename=False)[1];fast=build.change('FAST',build.review.original()[1:])[1]
 sources={'S15':s15,'CANCEL':cancel,'FAST_S15':fast,'BASE':build.parent(('PAIR','BLOCK'))[1],**{a:build.change(a,build.parent(('PAIR','BLOCK')))[1] for a in ('FAST','INDEX','FLUSH1')}}
 rows={}
 for name,cu in sources.items():
  p=O/('cost_'+name.lower()+'.cu');p.write_text(cu);rows[name]=dict(cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),flags=full(p));print(name,{f:(v['instructions'],v['registers']) for f,v in rows[name]['flags'].items()},flush=True)
 base=sources['BASE'];a=base.index('#pragma unroll\n      for (int j = 0; j < NT; ++j) {',base.index('auto unit_body'));b=base.index('    }\n    if constexpr (TRACE)',a);fill=base[a:b]
 header='''#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
__device__ __forceinline__ unsigned long long mark(){unsigned long long t;asm volatile("mov.u64 %0,%%clock64;":"=l"(t)::"memory");return t;}
template<int MODE,int PASS> __global__ void p_fill(const __nv_bfloat16* x,const __nv_bfloat16* inter,const int16_t* pairs,uint32_t* output,unsigned long long* stamps,int ne,int off,int nb,int pass_in){
 __shared__ int16_t pairbuf[256];int tid=threadIdx.x,lane=tid&31,warp=tid>>5,q=lane&3;constexpr int NT=1,NB=8,HD=4096;int pass=PASS<0?pass_in:PASS;
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
 source=header+fill+'\n}else{\n'+(D/'bfill.py.inc').read_text()+tail
 for mode in (0,1):
  for pas in (-1,0,1):source+=f'\ntemplate __global__ void p_fill<{mode},{pas}>(const __nv_bfloat16*,const __nv_bfloat16*,const int16_t*,uint32_t*,unsigned long long*,int,int,int,int);'
 p=O/'marked_fill.cu';p.write_text(source);sass,_=compile(p);probes=[]
 for name,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
  ins=instructions(b);marks=[i for i,(pc,o) in enumerate(ins) if 'CS2R' in o and 'SR_CLOCK' in o];assert len(marks)==2
  v=ins[marks[0]+1:marks[1]];op=ops(v);probes.append(dict(name=name,instructions=len(v),ops=op,integer_address_control=sum(op.get(k,0) for k in INT)))
 oldmath=json.loads((D.parent.parent/'issue_counts.json').read_text());math=next(v for v in oldmath['probes'] if 'p_mathILi1E' in v['name'] and v['segment']==1)
 r=dict(full_kernels=rows,marked_fill=probes,integer_classes=list(INT),decoder_math_unchanged=True,math_per16={k:v/16 for k,v in math['ops'].items()},normalization='B-fill happens once per expert/token-block segment: divide its cost by16 requested groups per warp tile,then by actual segment tiles L. Static path sets include empty/exception branches; no dynamic issue/stall attribution.',CANCEL='Same125regs and zero spills as S15; full static69 adds expert gathers,metadata loads,range/sign tests and duplicated FP64 tails. Ordinary finite eps takes the FP32 branch; FP64 text is not evidence it executes normally.',FAST='q=256*gs2/sqrt(eps); metadata(mul,eps,num) bounds runtime FP32 terms; zero norm guarded,sign in numerator; no runtime FP64 or exponent/sign branch',FLUSH1='Second consecutive flush after nv0 reset only adds zeros; one transition flush retained',INDEX='Hoist route validity/pass branches outside8 fills; one bounded32bit vector index per segment; all16 input vector reads retained')
 r['zero_rotation_search']=rotation_search()
 (D/'sass_cost.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(probes,indent=2))
def rotation_search():
 mask=0x81c081c0;rot=lambda x,s:((x<<s)|(x>>(32-s if s else 32)))&0xffffffff;ms=[rot(mask,i) for i in range(32)];choices=[]
 for x,y in itertools.combinations(range(1,32),2):
  if not(mask&ms[x] or mask&ms[y] or ms[x]&ms[y]):choices.append(((0,x,y),(~(mask|ms[x]|ms[y]))&0xffffffff))
 solutions=[]
 for ar,al in choices:
  for br,bl in choices:
   if al&bl:continue
   for x,y in itertools.combinations(range(1,32),2):
    if ms[x]&ms[y]==0 and ms[x]|ms[y]==al|bl:solutions.append([ar,br,[x,y]])
 return dict(mask=hex(mask),three_pure_rotation_choices_per_word=len(choices),solutions=solutions,scope='Restricted same-mask3 pure rotations per word plus2 from one LOP3-selected word,with zero rotation in both A and B. No solution saves one SHF by this layout alone; other decoders remain open.')
if __name__=='__main__':main()
