"""Exact lab-only wrapper rehearsal: forced S19 glue and packed-ready math."""
import builtins,hashlib,importlib.util,json,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(H),str(R/'tools')]
import torch,build_deletion,h200_lab
from lab_packet import read_packets
from sample_codec import unpack_samples

def main():
 build_deletion.main();payload=(H/'out/deletion.py').read_text();sha=hashlib.sha256(payload.encode()).hexdigest()
 wrapper,_=h200_lab.build([('deletion',sha[:16],payload)],'muse_prefill_layer',phase='import',import_budget=820,lab_only=True)
 path=H/'out/wrapper.py';path.write_text(wrapper)
 sp=importlib.util.spec_from_file_location('cf_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');scr=importlib.util.module_from_spec(sp);sp.loader.exec_module(scr)
 def ready_install(host,v):
  def ready_attention(qkv,d,state,plan):
   c=d['config'];T=c['tokens'];half=c['q_len']==1 and not c['window'];rb=d['_cf_ready'].view(torch.uint8)
   q=rb[:,:2048].view(torch.float16).reshape(T,16,64) if half else rb[:,:1024].view(torch.float8_e4m3fn).reshape(T,16,64)
   q=(q.float()*c['qk_scale_factor']).bfloat16();gate=d['_cf_ready'][:,1280:].float()
   if not gate.numel():gate=torch.zeros((T,1024),device=q.device)
   return host.attention_triton(q,gate,d,state)
  v['_cf_original_attention']=ready_attention
 m=types.ModuleType('cf_wrapper');m.__file__=str(path);emitted=[];archive=[]
 def emit(*args,**kwargs):
  line=kwargs.get('sep',' ').join(map(str,args))+kwargs.get('end','\n');emitted.append(line)
  if not line.startswith('LAB_PACKET '):builtins.print(*args,**kwargs)
 m.print=emit
 def install():
  m.CF_LOCAL_INSTALL=lambda host:scr.install(host,True);m.CF_LOCAL_NAMESPACE=ready_install;m.CF_LOCAL_ARCHIVE=lambda p:archive.append(json.loads(json.dumps(p)));original=m._lab_run_one
  def run(name,src,deadline,weights,state):return original(name,src+'\nLOCAL=True\nSEEDS=SEEDS[:1]\nROUNDS=2\n',time.monotonic()+3600,weights,state)
  m._lab_run_one=run
 m._CF_INSTALL=install;marker='\n_lab_import_start = ';assert wrapper.count(marker)==1
 exec(compile(wrapper.replace(marker,'\n_CF_INSTALL()'+marker,1),str(path),'exec'),m.__dict__)
 try:m.prepare(None);raise AssertionError('lab-only prepare returned')
 except RuntimeError as exc:assert 'lab-only host' in str(exc),exc
 tail=(''.join(emitted)+'P'*4000)[-16000:];records=read_packets(tail);assert len(records)==1 and records[0]['source_sha256']==sha[:16]
 p=unpack_samples(records[0]['packet']);assert p['complete'] and not p.get('transport_errors'),p.get('transport_errors');assert p['checks']['BASE']==p['checks']['AA']==12
 assert p['identities']['base_sha256']==p['base_sha256'] and p['seeds']==p['seed_ids']
 for a,b in zip(p['paired'],archive[0]['paired']):assert a['round_ns']==b['round_ns'] and a['telemetry']==b['telemetry']
 wire=max(len(x.encode()) for x in emitted if x.startswith('LAB_PACKET '))
 out=dict(payload_sha256=sha,wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),checks=p['checks'],cases=len(p['paired']),wire_bytes=wire,tail_with_4000_suffix_pass=True,lossless_round_and_telemetry_roundtrip=True,standins='Unchanged S19 forced glue; ready byte layout decoded into portable all-key Triton attention. Native oracle CUDA compiled separately for sm90a; no timing/native-execution claim from stand-ins.')
 (H/'screen_forced.json').write_text(json.dumps(out,indent=2)+'\n');(H/'out/local_packet.json').write_text(json.dumps(p)+'\n');print('PASS',out)
if __name__=='__main__':main()
