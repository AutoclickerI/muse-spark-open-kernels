"""Run the instrumented and widened producers through the unchanged task gate."""
import hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode')]
from local_eval import load_module,capture,copy_into,undefined_names,call_signatures,hopper_compile
m=load_module('producer_phase_lab',ROOT/'prof/lab/verify8_producer_phases.py')
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *a,**k:None
rows=[]
for wide in (False,True):
    m.phase_ext(wide)
    for st,kt in ((3,64),(4,64),(3,128),(4,128)):
        for ctx,dist,scale,ns in ((10,'normal',1.,17),(4099,'coherent',1.,7),(3077,'edges',1./64.,5)):
            d=ref.generate_input(2,ctx,'tp8','full',812,q_len=8,page=1,dist=dist,kv_scale=scale)
            setting=(ns,st,kt)
            def fn(d):return m.phase_run(d,setting,False)[0]
            graph,out=capture(fn,d)
            fresh=ref.generate_input(2,ctx,'tp8','full',813,q_len=8,page=1,dist=dist,kv_scale=scale)
            copy_into(d,fresh);graph.replay();torch.cuda.synchronize()
            ok,msg=ref.check_implementation(d,ref.collect(d,out));assert ok,(wide,setting,ctx,msg)
            traced,clocks=m.phase_run(d,setting,True);torch.cuda.synchronize()
            tok,tmsg=ref.check_implementation(d,ref.collect(d,traced));assert tok,tmsg
            relative=float((out.float()-traced.float()).norm()/out.float().norm().clamp_min(1e-12))
            assert relative<2.e-5,relative
            assert torch.isfinite(clocks.float()).all()
            rows.append(dict(wide=wide,setting=setting,ctx=ctx,dist=dist,scale=scale,gate=ok,trace_gate=tok,trace_relative=relative,msg=msg,graph=True))
            print(rows[-1],flush=True)
for B in (16,32):
    d=ref.generate_input(B,80000,'tp8','full',967+B,q_len=8,page=1,spread='agentic',dist='coherent')
    for wide,setting in ((False,(17,3,128)),(True,(17,3,128)),(True,(34,3,64))):
        out,_=m.phase_run(d,setting,False,wide);torch.cuda.synchronize();expected=m.gate_prepare(d)
        ok,msg,err=m.gate_cached(d,out,expected);assert ok,msg
        official,msg2=ref.check_implementation(d,ref.collect(d,out));assert official,msg2
        us,out,g=m.timed(lambda x:m.phase_run(x,setting,False,wide)[0],d,torch.empty(128<<20,device='cuda',dtype=torch.uint8)) if not wide else m.timed(lambda x:m.phase_run(x,setting,False,True)[0],d,torch.empty(128<<20,device='cuda',dtype=torch.uint8))
        ok,msg,err=m.gate_cached(d,out,expected);assert ok,msg
        official,msg2=ref.check_implementation(d,ref.collect(d,out));assert official,msg2
        rows.append(dict(B=B,wide=wide,setting=setting,ctx=80000,gate=ok,official_gate=official,error=err,local_us=us))
        print(rows[-1],flush=True)
result=dict(payload_sha256=hashlib.sha256((ROOT/'prof/lab/verify8_producer_phases.py').read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),cc=torch.cuda.get_device_capability(),checks=rows)
(ROOT/'problems/muse_attention_decode/dev/producer_phases/validation_local.json').write_text(json.dumps(result,indent=2)+'\n')
