"""Exact S23 packed rehearsal: every arm, protocol counts and the actual receipt tail.

Stand-ins are P10's night/screen_s22.py chain (S22 screen under the night NF/TMA stand-ins); the
incumbent arm (S22 phase 1, no P10 helpers) gets the S22 screen alone, forced as that chain forces it.
"""
import builtins,hashlib,json,sys,time,types
import compose
H=compose.H;R=compose.R;NIGHT=R/'problems/muse_bf16_proj/dev/public_ports/night'

def standins():
 def load(name,path):
  m=compose.load(path,name);sys.modules[name]=m;return m
 local=load('ng_candidate_standins',NIGHT/'local_standins.py');s22=load('ng_s22_screen',H.parent/'s22/screen.py')
 local.prior.s19=s22;local.prior.calls=local.calls
 def install(v):
  if hasattr(v,'pp_prepare'):local.install(v,True)
  else:s22.install(v,True)
 return install

def main():
 rep=compose.load(H/'report.py','s23_report');meta=json.loads((H/'manifest.json').read_text());wrapper=H/'generated/wrapper.py';source=wrapper.read_text()
 assert hashlib.sha256(source.encode()).hexdigest()==meta['wrapper_sha256']
 install_arm=standins();m=types.ModuleType('submission');m.__file__=str(wrapper);emitted=[]
 def emit(*args,**kwargs):
  emitted.append(kwargs.get('sep',' ').join(map(str,args))+kwargs.get('end','\n'));return builtins.print(*args,**kwargs)
 m.print=emit
 def install():
  m.S16_LOCAL_INSTALL=install_arm;run=m._lab_run_one
  m.S21_LOCAL_ARCHIVE=lambda raw:(H/'generated/rehearsal_raw.json').write_text(json.dumps(raw)+'\n')
  def checked(name,src,deadline,weights,state):return run(name,src,time.monotonic()+3600,weights,state)
  m._lab_run_one=checked
 m._S23_INSTALL=install;marker='\n_lab_import_start = ';assert source.count(marker)==1;t0=time.monotonic();exec(compile(source.replace(marker,'\n_S23_INSTALL()'+marker,1),str(wrapper),'exec'),m.__dict__)
 try:m.prepare(None);raise AssertionError('lab-only prepare returned')
 except RuntimeError as exc:assert 'lab-only host' in str(exc),exc
 sys.path.insert(0,str(R/'tools'));from lab_packet import paired_evidence,read_packets
 from promotion import _protocol_arm_valid
 tail=(''.join(emitted)+'P'*4000)[-16000:];records=read_packets(tail);assert len(records)==1 and records[0]['source_sha256']==meta['payload_sha256'][:16]
 e=paired_evidence(tail,'S23');assert e is not None;raw=rep.decode(e)
 assert all(_protocol_arm_valid(raw,meta,n,64) for n in ('A','A2',*meta['arms']))
 def compact(x):
  if isinstance(x,float):return round(x,6)
  if isinstance(x,list):return [compact(v) for v in x]
  if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
  return x
 archived=compact(json.loads((H/'generated/rehearsal_raw.json').read_text()));assert rep.codec.s21_unpack_evidence(e)==archived
 out=dict(wrapper_sha256=meta['wrapper_sha256'],payload_sha256=meta['payload_sha256'],source_sha256=meta['sha256'],complete=raw['complete'],failures=raw['failures'],TASK=raw['task_checks'],nearby=raw['nearby_task_checks'],
  arm_TASK={n:s['task_checks'] for n,s in raw['arm_status'].items()},archive_round_trip_exact=True,tail_with_4000_suffix_pass=True,
  wire_bytes=max(len(x.encode()) for x in emitted if x.startswith('LAB_PACKET ')),import_s=round(time.monotonic()-t0,1),host_prepare_s=raw.get('host_prepare_s'))
 assert out['complete'] and not out['failures'] and out['TASK']==meta['expected_TASK_checks'] and out['nearby']==260
 (H/'generated/lab_forced_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
