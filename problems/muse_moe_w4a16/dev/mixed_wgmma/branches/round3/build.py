"""Build one expert-major tail packet against the exact S14 and CTA2 controls."""
import ast
import hashlib
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;B=H.parent;P=B/'round2'
sys.path.insert(0,str(B))
import importlib.util
spec=importlib.util.spec_from_file_location('p3b_r2_builder',P/'build.py');r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
from light_build import value

def function(s,name):
 return ast.get_source_segment(s,[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name][-1])

def cuda(cu,deep):
 a=deep.index('template<int NT,bool F,int CS,int S>');b=deep.index('template<int MAX,bool F,int CS>__global__',a)
 t3='namespace qt3 {using namespace oc;\n'+deep[a:b]+'\n}\n'
 a=cu.index('template<bool META>__global__');b=cu.index('// SPLIT blocks',a)
 route=cu[a:b].replace('route_live(', 'route(').replace('int*T8,int M,int NT,int start){','int*T8,int M,int NT,int start,int*work){')
 # Add one compact-list scan to the existing all-CTA routing barrier. The
 # heavy last <=64-token chunk is included for nearby M>256 as well.
 end=route.rfind('}')
 route=route[:end]+'''
 __syncthreads();
 if(t<32){int e=ORDER[t],n=C[e],rem=n%256;
  int eligible=(n>8&&n<=64)||(n>256&&rem>0&&rem<=64),scan=eligible;
  #pragma unroll
  for(int d=1;d<32;d*=2){int v=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=v;}
  if(eligible)work[3+scan-1]=e;
  if(t==31)work[2]=scan;
  if(t==0){work[0]=0;work[1]=0;}
 }
'''+route[end:]
 return cu+'\n'+t3+'\n'+(H/'tail.cu.inc').read_text().replace('TAIL_ROUTE',route)

def plan(s):
 f=function(s,'reuse_plan').replace('def reuse_plan(','def tail_plan(').replace('arm in (1,2)','False')
 f=f.replace('a1=torch.empty(', 'a1=torch.empty(') # layout code is folded out below
 f=f.replace(" a1=torch.empty((32,(m+63)//64,256,128,4),device=dev,dtype=torch.int32) if False else x\n a2=torch.empty_like(a1) if False else x", " work=torch.empty(35,device=dev,dtype=torch.int32);a1=work;a2=x")
 f=f.replace('mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)', 'mod.tail_route(rp,work.data_ptr(),m,start,torch.cuda.current_stream().cuda_stream)')
 a=f.index(' def body():');b=f.index(' def finish():',a)
 f=f[:a]+''' def body():
  _=owners;parent=torch.cuda.current_stream();fork.record(parent)
  for i,side in enumerate(sides[:4]):
   side.wait_event(fork)
   with torch.cuda.stream(side):
    if i==0:
     mod.tail_run(p13,work.data_ptr(),0 if arm==2 else 1,3 if arm==4 else 2,1,m,sms*2,side.cuda_stream)
     mod.tail_run(p2,work.data_ptr(),0 if arm==2 else 1,3 if arm==4 else 2,0,m,sms*2,side.cuda_stream)
    else:
     mod.oc_run(p13,(128,192,256)[i-1],1,1,sms*4,m,side.cuda_stream);mod.oc_run(p2,(128,192,256)[i-1],0,1,sms*4,m,side.cuda_stream)
    joins[i].record()
  mod.pmz_run(pn,1,8,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)
  for join in joins[:4]:parent.wait_event(join)
'''+f[b:]
 assert 'work=torch.empty' in f
 return f

def main():
 r2.main();s=(P/'payload.py').read_text();old=value(s,'OP_CUDA');cu=cuda(old[3],old[4]);cpp=value(s,'OP_CPP')[3]+'''void tail_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void tail_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> tail_info(int64_t,int64_t,int64_t);'''
 body=s[:s.index('OP_CUDA=')]+'\nTAIL_CUDA='+repr(cu)+'\nTAIL_CPP='+repr(cpp)+'\nNF_OPTS={}\n'+function(s,'reuse_plan')+'\n'+plan(s)+'\n'
 assert 'OP_MODS[OP_ARM]' in function(s,'nf_module')
 body+='\n'.join(function(s,n) for n in ('op_prepare','nf_module','op_call','op_error','op_pair'))+'\n'+(H/'runtime.py.inc').read_text()
 compile(body,'<round3>','exec');(H/'tail_payload.py').write_text(body);(H/'tail.cu').write_text(cu)
 for arm,name in ((2,'static'),(3,'queue'),(4,'queue_s3')):
  screen='# okc: problem=muse_moe_w4a16\ndef lab_cold_l2():pass\n'+body+f'\nSCREEN_ARM={arm}\n'+'''def prepare(weights):
    op_build()
    w={k:(v[:32].contiguous() if k.startswith(('w13','w2')) else v) for k,v in weights.items()}
    w['post_expert_norm']=w['post_norm_w']
    return op_prepare(w)
def kernel_fn(data,state):return op_call(data,state,SCREEN_ARM)
'''
  (H/('screen_'+name+'.py')).write_text(screen)
 m=dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),CTA2_prefix_sha256=hashlib.sha256(old[3].encode()).hexdigest(),base='S14 f1c77ce6',arms=['S14','CTA2','TAIL_STATIC','TAIL_QUEUE','TAIL_QUEUE_S3'])
 (H/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m)

if __name__=='__main__':main()
