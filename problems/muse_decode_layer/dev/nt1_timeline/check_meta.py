"""End-to-end cluster MoERouteMeta producer plus S11 consumer, fresh graph inputs and global expert ranges."""
import ast,hashlib,json,linecache,runpy
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text();name='<meta fixture>';linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={};exec(compile(src,name,'exec'),n)
c=runpy.run_path(str(DEV/'meta_cell.py'));tree=ast.parse((ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text())
fields={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and len(node.targets)==1 and getattr(node.targets[0],'id','') in ('GLUE_CL_CPP','GLUE_CL_CUDA')}
tag=hashlib.sha256(fields['GLUE_CL_CUDA'].encode()).hexdigest()[:12]
prod=load_inline(name='nt1_meta_producer_'+tag,cpp_sources=fields['GLUE_CL_CPP'],cuda_sources=fields['GLUE_CL_CUDA'],functions=['route_cl'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr'],verbose=False)
w=n['rg_generate_weights'](5353);w['post_norm_w']=w['post_expert_norm'];old=n['nt1_banks'](w)
st={k:old[k] for k in ('pnw','gs13','gs2','sms')};st['banks']={'lop3':old['bank']};c['metam_bind'](st,{8:101,16:101,32:101})

def route(pr,rs,meta,arrive):
 p,s=pr;m=p.shape[1];x=torch.empty((m,4096),device=p.device,dtype=torch.bfloat16);ids=torch.empty((m,8),device=p.device,dtype=torch.int32);lids=torch.empty_like(ids);tw=torch.empty((m,8),device=p.device)
 prod.route_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),rs['bias'].data_ptr(),rs['pre'].data_ptr(),rs['start'].data_ptr(),x.data_ptr(),ids.data_ptr(),tw.data_ptr(),lids.data_ptr(),meta.data_ptr(),arrive.data_ptr(),m,8,256,torch.cuda.current_stream().cuda_stream)
 d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0},route_meta=meta)
 return c['metam_experts'](d,st,101),d,ids

checks=0;worst=0.;vsinc=0.
for m in (8,16,32):
 for start in (0,96,224):
  pr,rs=n['routed_fixture'](w,m,3941+m);rs['start'].fill_(start)
  meta=torch.empty(736,device='cuda',dtype=torch.int32);arrive=torch.zeros(1,device='cuda',dtype=torch.int32)
  route(pr,rs,meta,arrive);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out,d,ids=route(pr,rs,meta,arrive)
  for draw in range(6):
   fresh,_=n['routed_fixture'](w,m,6231+draw*13+m);pr[0].copy_(fresh[0]);g.replay();torch.cuda.synchronize()
   ok,err,row=n['task_check'](out,n['nt1_fp32'](d,w));inc=n['nt1_call'](d,old,'INC');torch.cuda.synchronize()
   rel=float((out.float()-inc.float()).norm()/inc.float().norm().clamp_min(1e-15));assert ok and rel<2e-3,(m,start,draw,err,row,rel)
   assert int(meta[0])==1 and int(meta[1])==m and int(meta[2])==start and int(arrive[0])==0
   n['route_gate'](ids,n['route_reference'](pr,rs)[2],start)
   worst=max(worst,err);vsinc=max(vsinc,rel);checks+=1
  print('META PASS',m,start,flush=True)
result=dict(fresh_graphs=checks,worst_task_l2=worst,max_rel_vs_inc=vsinc,cell_sha256=hashlib.sha256((DEV/'meta_cell.py').read_bytes()).hexdigest(),producer_sha256=hashlib.sha256(fields['GLUE_CL_CUDA'].encode()).hexdigest())
(DEV/'meta_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
