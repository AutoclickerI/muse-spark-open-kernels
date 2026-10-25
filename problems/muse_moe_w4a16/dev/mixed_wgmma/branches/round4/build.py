"""Build TMA/raw -> decode/A-fragment -> paired RS-WGMMA pipeline."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;B=H.parent;P=B/'round2'
sys.path.insert(0,str(B))
spec=importlib.util.spec_from_file_location('p3b_r2_builder',P/'build.py');r2=importlib.util.module_from_spec(spec);spec.loader.exec_module(r2)
from light_build import value
import cluster_build

def fun(s,n):return ast.get_source_segment(s,[a for a in ast.parse(s).body if isinstance(a,ast.FunctionDef) and a.name==n][-1])

def main():
 r2.main();src=(P/'payload.py').read_text();control=value(src,'OP_CUDA')[3];cpp=value(src,'OP_CPP')[3]
 a=control.index('template<int SPLIT>__global__ void finish');b=control.index('}  // namespace mwbr',a)
 finish=control[a:b].replace('int M,int start){','int M,int start,const int*C,const float*HP){')
 old='sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];'
 assert old in finish
 finish=finish.replace(old,'if(C[e]>16){for(int j=0;j<8;++j)sum+=HP[(size_t)p*256+l+j*32];}else '+old)
 cu=control+'\n'+(H/'pipeline.cu.inc').read_text().replace('WS_MMA',cluster_build.mma()).replace('WS_FINISH',finish)
 extra='''void ws_gather(std::vector<int64_t>,int64_t,int64_t,int64_t);std::vector<int64_t> ws_info(int64_t,int64_t,int64_t);void ws_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void ws_finish(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);'''
 prefix=src[:src.index('OP_CUDA=')]
 plan=fun(src,'reuse_plan').replace('def reuse_plan(', 'def ws_plan(')
 plan=plan.replace('arm in (1,2)','False').replace('a1=torch.empty((32,(m+63)//64,256,128,4),device=dev,dtype=torch.int32) if False else x\n a2=torch.empty_like(a1) if False else x', 'a1=torch.empty_like(act);a2=torch.empty((cap,256),device=dev,dtype=torch.float32);h13[3]=a1;p13=[t.data_ptr() for t in h13]')
 a=plan.index(' def body():');b=plan.index(' def finish():',a)
 plan=plan[:a]+''' def body():
  _=owners;parent=torch.cuda.current_stream();pv=list(p13);pv[3]=x.data_ptr();mod.ws_gather(pv,a1.data_ptr(),m,parent.cuda_stream);fork.record(parent)
  for side,join,n in zip(sides,joins,(64,128,192,256)):
   side.wait_event(fork)
   with torch.cuda.stream(side):
    mod.ws_run(p13,base['packet13'].data_ptr(),a2.data_ptr(),n,arm==3,1,m,sms,side.cuda_stream)
    mod.ws_run(p2,base['packet2'].data_ptr(),a2.data_ptr(),n,arm==3,0,m,sms,side.cuda_stream)
    join.record()
  sides[4].wait_event(fork)
  with torch.cuda.stream(sides[4]):mod.br_run(pl13,70,sms*4,m,sides[4].cuda_stream);mod.br_run(pl2,71,sms*4,m,sides[4].cuda_stream);joins[4].record()
  mod.pmz_run(pn,1,8,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)
  for join in joins:parent.wait_event(join)
'''+plan[b:]
 plan=plan.replace('mod.br_finish(fp,m,start,4,torch.cuda.current_stream().cuda_stream)', 'mod.ws_finish(fp,c.data_ptr(),a2.data_ptr(),m,start,torch.cuda.current_stream().cuda_stream)')
 body=prefix+'\nWS_CUDA='+repr(cu)+'\nWS_CPP='+repr(cpp+extra)+'\nNF_OPTS={}\n'+fun(src,'reuse_plan')+'\n'+plan+'\n'
 body+='\n'.join(fun(src,n) for n in ('nf_module','op_call','op_error','op_pair'))+'\n'+(H/'runtime.py.inc').read_text()
 compile(body,'<WS lab>','exec');(H/'ws_payload.py').write_text(body);(H/'ws.cu').write_text(cu)
 for arm,name in ((2,'barrier'),(3,'pipe')):
  screen='# okc: problem=muse_moe_w4a16\ndef lab_cold_l2():pass\n'+body+f'\nSCREEN_ARM={arm}\n'+'''def prepare(weights):
    op_build()
    w={k:(v[:32].contiguous() if k.startswith(('w13','w2')) else v) for k,v in weights.items()}
    w['post_expert_norm']=w['post_norm_w']
    return op_prepare(w)
def kernel_fn(data,state):return op_call(data,state,SCREEN_ARM)
'''
  (H/('screen_'+name+'.py')).write_text(screen)
 m=dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),CTA2_prefix_sha256=hashlib.sha256(control.encode()).hexdigest(),base='S14 f1c77ce6',arms=['S14','CTA2','WS_TILE_BARRIER','WS_PIPE'])
 (H/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m)

if __name__=='__main__':main()
