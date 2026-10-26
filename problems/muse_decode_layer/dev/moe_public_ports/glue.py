"""Force S19 and F Hopper launch glue without executing Hopper instructions."""
import importlib.util,json,re
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('public_glue',H/'out/public_moe.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
base=m.module('glueS19',m.BUNDLE['base']);lee=m.module('glueL',m.BUNDLE['lee']);calls=[]
class Native:
 smem_optin=1<<22
 def resident_blocks(self,*a):return 2
 def smem_needed(self,*a):return 0
 def epi_info(self,*a):return [2]
 def pmz_init(self):return 1<<22
 def pmz_smem(self,*a):return 0
 def pmz_resident(self,*a):return 1
 def ch_info(self,*a):return [1]
 def run(self,ptr,M,lo,stages,grid,stream,flags):assert len(ptr)==13 and M<=32;calls.append(['small',M,flags])
 def br_route(self,ptr,*a):assert len(ptr)==9;calls.append(['route',len(ptr)])
 def ch_run(self,ptr,*a):assert len(ptr)==13;calls.append(['heavy',len(ptr)])
 def br_run(self,ptr,*a):assert len(ptr)==13;calls.append(['light',len(ptr)])
 def pmz_run(self,ptr,*a):assert len(ptr)==12;calls.append(['native',len(ptr)])
 def mt_finish(self,ptr,*a):assert len(ptr)==7;calls.append(['finish',len(ptr)])
fake=Native()
for n in ('smallm_module','rvc_nfidx0_module','rvc_nbase1_module','rvc_nbase2_module','p3b_module','mt_module','mt_pmz_module'):setattr(base,n,lambda:fake)
dev='cuda';w=dict(w13=torch.zeros((32,8192,2048),dtype=torch.uint8,device=dev),w13_sf=torch.full((32,8192,256),.5,dtype=torch.float8_e4m3fn,device=dev),w2=torch.zeros((32,4096,2048),dtype=torch.uint8,device=dev),w2_sf=torch.full((32,4096,256),.5,dtype=torch.float8_e4m3fn,device=dev),w13_gs=torch.ones(32,device=dev),w2_gs=torch.ones(32,device=dev),post_norm_w=torch.zeros(4096,dtype=torch.bfloat16,device=dev))
sb=m.s19_prepare(base,w)
for M in (4,)+m.MS:
 d=dict(hidden_states=torch.zeros((M,4096),dtype=torch.bfloat16,device=dev),topk_ids=torch.zeros((M,8),dtype=torch.int32,device=dev),topk_weights=torch.ones((M,8),device=dev)/8,config=dict(local_start=0))
 p=m.s19_plan(base,d,sb);m.run_plan(p)
 for _,fn in p.get('phase_steps',[]):fn()
 for _,fn in p['steps']:fn()
class F:
 def port_f(self,ptr,M,lo,stream):assert len(ptr)==12 and M<=32 and lo==0 and isinstance(stream,int);calls.append(['F',M])
st={k:torch.empty(1,device=dev) for k in ('w1','w2f','g1','g2','nw')};d['hidden_states']=d['hidden_states'][:8];d['topk_ids']=d['topk_ids'][:8];d['topk_weights']=d['topk_weights'][:8];m.run_plan(lee.lee_plan(d,st,F(),'F'))
passes=[l.split()[1:] for l in (H/'out/screen.log').read_text().splitlines() if l.startswith('PASS ')];assert len(passes)==50,len(passes)
(H/'local_validation.json').write_text(json.dumps(dict(passed=True,real_eager_graph_checks=passes,standin_calls=calls,limits='X actual sm120 mma.sync emulation; F and exact S19 Hopper ABI stand-ins; all four changed sm90a units compiled separately'),indent=2)+'\n')
print('PASS',len(passes),'real arms,',len(calls),'Hopper glue calls')
