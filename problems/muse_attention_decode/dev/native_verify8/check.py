"""Gate the native path via its matching warpgroup shim and force its Hopper launcher."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode')]
from local_eval import load_module,capture,copy_into
m=load_module('native_verify8_adapter',ROOT/'problems/muse_decode_layer/dev/verify8_native_wgmma.py')
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *a,**kw:None
rows=[]
for st in (4,8):
 for ctx,dist,page,scale,ns in ((10,'normal',1,1.,17),(3077,'edges',1,1./64.,7),(4099,'coherent',64,1.,5)):
  data=ref.generate_input(2,ctx,'tp8','full',823,q_len=8,page=page,dist=dist,kv_scale=scale)
  setting=(ns,st)
  def fn(d):return m.native_attention(d['qkv'],d,setting,force_hopper=True)[0]
  g,out=capture(fn,data)
  fresh=ref.generate_input(2,ctx,'tp8','full',824,q_len=8,page=page,dist=dist,kv_scale=scale)
  copy_into(data,fresh);g.replay();torch.cuda.synchronize()
  ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
  traced,clock=m.native_attention(data['qkv'],data,setting,clocks=True,force_hopper=True);torch.cuda.synchronize()
  tok,tmsg=ref.check_implementation(data,ref.collect(data,traced));assert tok,tmsg
  equal=bool(torch.equal(out,traced));assert equal
  rows.append(dict(stages=st,ctx=ctx,dist=dist,page=page,scale=scale,ns=ns,gate=ok,clock_gate=tok,clock_bitwise=equal,msg=msg,graph=True,forced_hopper_glue=True))
  print(rows[-1],flush=True)
for B in (1,16,32):
 data=ref.generate_input(B,80000,'tp8','full',917+B,q_len=8,page=1,spread='agentic' if B>1 else 'fixed',dist='coherent')
 for count in (0,1,4):
  if count:
   partial=torch.randn((count,B*8,2304),device='cuda',dtype=torch.float32)
   data['qkv']=partial.sum(0).to(torch.bfloat16).view(B,8,2304);qkv=(partial,count)
  else:qkv=data['qkv']
  def fn(d):return m.attention_verify8_native(qkv,d).view(B,8,1024)
  g,out=capture(fn,data)
  if count:
   partial.add_(.03125);data['qkv']=partial.sum(0).to(torch.bfloat16).view(B,8,2304)
  else:data['qkv'].add_(.03125)
  g.replay();torch.cuda.synchronize()
  ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
  rows.append(dict(B=B,ctx=80000,count=count,gate=ok,msg=msg,graph=True,forced_hopper_glue=True));print(rows[-1],flush=True)
result=dict(adapter_sha256=hashlib.sha256((ROOT/'problems/muse_decode_layer/dev/verify8_native_wgmma.py').read_bytes()).hexdigest(),cuda_sha256=hashlib.sha256(m.NATIVE_CUDA.encode()).hexdigest(),native_hopper=False,gpu=torch.cuda.get_device_name(),checks=rows)
(ROOT/'problems/muse_attention_decode/dev/native_verify8/checks_local.json').write_text(json.dumps(result,indent=2)+'\n')
