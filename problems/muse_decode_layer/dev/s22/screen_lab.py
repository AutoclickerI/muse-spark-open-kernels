"""Exact S22 packed rehearsal, all per-cell arms, protocol counts and actual receipt tail."""
import builtins,hashlib,json,sys,time,types
from pathlib import Path
import compose
H=compose.H

def main():
 forced='--forced' in sys.argv;scr=compose.load(H/'screen.py','s22_screen');rep=compose.load(H/'report.py','s22_report');meta=json.loads((H/'manifest.json').read_text());wrapper=H/'generated/wrapper.py';source=wrapper.read_text()
 assert hashlib.sha256(source.encode()).hexdigest()==meta['wrapper_sha256']
 m=types.ModuleType('submission');m.__file__=str(wrapper);emitted=[]
 def emit(*args,**kwargs):
  emitted.append(kwargs.get('sep',' ').join(map(str,args))+kwargs.get('end','\n'));return builtins.print(*args,**kwargs)
 m.print=emit
 def install():
  m.S16_LOCAL_INSTALL=lambda v:scr.install(v,forced);run=m._lab_run_one
  m.S21_LOCAL_ARCHIVE=lambda raw:(H/'generated/rehearsal_raw.json').write_text(json.dumps(raw)+'\n')
  def checked(name,src,deadline,weights,state):return run(name,src,time.monotonic()+3600,weights,state)
  m._lab_run_one=checked
 m._S22_INSTALL=install;marker='\n_lab_import_start = ';assert source.count(marker)==1;t0=time.monotonic();exec(compile(source.replace(marker,'\n_S22_INSTALL()'+marker,1),str(wrapper),'exec'),m.__dict__)
 try:m.prepare(None);raise AssertionError('lab-only prepare returned')
 except RuntimeError as exc:assert 'lab-only host' in str(exc),exc
 sys.path.insert(0,str(compose.R/'tools'));from lab_packet import paired_evidence,read_packets
 from promotion import _protocol_arm_valid
 tail=(''.join(emitted)+'P'*4000)[-16000:];records=read_packets(tail);assert len(records)==1 and records[0]['source_sha256']==meta['payload_sha256'][:16]
 e=paired_evidence(tail,'S22');assert e is not None;raw=rep.decode(e)
 assert all(_protocol_arm_valid(raw,meta,n,64) for n in ('A','A2',*meta['arms']))
 for name,spec in meta['arms'].items():
  if spec['kind'] in ('S19_cell','prior_port'):assert len(raw['cell_ratios'][spec['cell']][name])==64
 def compact(x):
  if isinstance(x,float):return round(x,6)
  if isinstance(x,list):return [compact(v) for v in x]
  if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
  return x
 archived=compact(json.loads((H/'generated/rehearsal_raw.json').read_text()));assert rep.codec.s21_unpack_evidence(e)==archived
 out=dict(wrapper_sha256=meta['wrapper_sha256'],payload_sha256=meta['payload_sha256'],source_sha256=meta['sha256'],complete=raw['complete'],failures=raw['failures'],TASK=raw['task_checks'],nearby=raw['nearby_task_checks'],
  per_cell_samples={k:{n:len(v) for n,v in cell.items()} for k,cell in raw['cell_ratios'].items()},sampling_plan_admitted=True,archive_round_trip_exact=True,tail_with_4000_suffix_pass=True,
  wire_bytes=max(len(x.encode()) for x in emitted if x.startswith('LAB_PACKET ')),import_s=round(time.monotonic()-t0,1))
 assert out['complete'] and not out['failures'] and out['TASK']==meta['expected_TASK_checks'] and out['nearby']==260
 (H/'generated/lab_forced_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
