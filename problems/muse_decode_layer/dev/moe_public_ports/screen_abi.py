"""Real ABI GEMMs and FP32 finalizer against TASK on the local GPU."""
import importlib.util,json
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('moe_abi_screen',H/'out/public_moe_abi.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
op=m.module('AbiOlocal',m.BUNDLE['abi_operator']);ck=m.module('AbiTASKlocal',m.BUNDLE['checker']);final=m.abi_final_module();w=ck.generate_weights(275163217);bank=op.prepare(w);checks=[]
for M,dist in ((33,'normal'),(128,'normal'),(128,'skewed'),(256,'normal'),(257,'normal')):
 d=ck.generate_input(bs=M,sharding='ep8',seed=276631991+M,dist=dist);x=d['hidden_states'];ids=d['topk_ids'];tw=d['topk_weights'];cnt=torch.stack([(ids==e).sum() for e in range(32)]).int();pad=(cnt+31)//32*32;off=torch.cat((torch.zeros(1,device='cuda',dtype=torch.int32),pad.cumsum(0).int()));slots=torch.empty((32,M),device='cuda',dtype=torch.int32);pos=torch.full_like(ids,-1)
 for e in range(32):
  pairs=torch.where(ids.reshape(-1)==e)[0].int();slots[e,:len(pairs)]=pairs;pos.reshape(-1)[pairs.long()]=torch.arange(len(pairs),device='cuda',dtype=torch.int32)
 cap=M*8+4096;act=torch.empty((cap,4096),device='cuda',dtype=torch.bfloat16);y=torch.empty((cap,4096),device='cuda');ss=torch.empty((cap,64),device='cuda');out=torch.empty_like(x)
 h=[None,None,None,x,cnt,off,None,None,slots,w['post_norm_w'],act,y,ss];owners=(h,None,None,ids,tw,pos,None,None,None);fp=[off,ids,tw,pos,y,ss,out];ptr=[t.data_ptr() for t in fp]
 def finish():final.abi_finish(ptr,M,torch.cuda.current_stream().cuda_stream);return out
 plan=dict(owners=owners,steps=[('route',lambda:None)],phase_steps=[('finish',finish)],out=out)
 p=op.abi_plan(d,bank,plan,'both');ref=ck.ref_kernel(d,w);yy=m.run_plan(p);torch.cuda.synchronize();ok,msg=ck.compare(yy,ref,ck.REL_L2_TOL);assert ok,(M,dist,'eager',msg)
 g,yy=m.capture(lambda:m.run_plan(p));g.replay();g.replay();torch.cuda.synchronize();ok,msg=ck.compare(yy,ref,ck.REL_L2_TOL);assert ok,(M,dist,'graph',msg)
 checks.append([M,dist]);print('ABI_PASS',M,dist,flush=True);del g,p,plan
(H/'abi_local_validation.json').write_text(json.dumps(dict(passed=True,eager_and_graph=checks,contracts='S19 sorted BF16 ACT, FP32 Y,64 FP32 SS tiles; full reference TASK; native finalizer'),indent=2)+'\n')
