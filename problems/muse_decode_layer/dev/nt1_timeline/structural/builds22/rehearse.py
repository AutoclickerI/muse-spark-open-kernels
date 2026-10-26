"""Local packed-harness rehearsal, explicitly two seeds/two rounds; no H200 claims.

Normal/forced complete TASK+12-shape screens and 1350 leaf gates are separate.
The transported H200 payload stays64x16 with all nearby cases. A marked synthetic
64x16 archive stresses only codec capacity; it is never a performance result.
"""
import ast,base64,copy,hashlib,importlib.util,json,linecache,lzma,random,struct,sys,time
from pathlib import Path
import torch
from sample_codec import p3a_pack_vectors,p3a_unpack_vectors,p3a_orders
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_builds22';sys.path.insert(0,str(R/'tools'));import lab_packet,promotion
sp=importlib.util.spec_from_file_location('p3a22_screen',D/'screen.py');screen=importlib.util.module_from_spec(sp);sp.loader.exec_module(screen)
sp=importlib.util.spec_from_file_location('p3a22_codec',R/'problems/muse_decode_layer/dev/s21/evidence.py');codec=importlib.util.module_from_spec(sp);sp.loader.exec_module(codec)
meta=json.loads((D/'lab_manifest.json').read_text());p=R/meta['payload'];source=p.read_text();assert hashlib.sha256(source.encode()).hexdigest()==meta['payload_sha256']
buf=[]
def cold():
 if not buf:buf.append(torch.empty(1<<30,device='cuda',dtype=torch.uint8))
 buf[0].view(torch.int64).sum()
def archive(raw):(O/'rehearsal_raw.json').write_text(json.dumps(raw)+'\n')
ns={'__name__':'p3a22_rehearsal','S16_LOCAL_INSTALL':lambda m:screen.install(m,True),'S21_LOCAL_ARCHIVE':archive,'lab_cold_l2':cold}
f='<p3a22-exact-packed>';linecache.cache[f]=(len(source),None,source.splitlines(True),f);exec(compile(source,f,'exec'),ns)
# Explicit local screen plan. These assignments never touch the immutable payload.
assert ns['S21_PROTOCOL']==meta['measurement_protocol'] and len(ns['AB_SEEDS'])==64
ns['AB_SEEDS']=(1779003,1783739);ns['S17_WEIGHT_SEED']=1775353;ns['S17_ROUTER_SEED']=1775355;ns['S21_PROTOCOL']=dict(meta['measurement_protocol'],seed_count=2,rounds=2)
ns['S17_NEARBY_CASES']=[ns['S17_NEARBY_CASES'][0],next(v for v in ns['S17_NEARBY_CASES'] if v.get('batch',1)*v.get('q_len',1)>32)]
(O/'rehearsal_raw.json').unlink(missing_ok=True)
t=time.monotonic();lines=ns['lab'](t+1800,None,None);raw=json.loads((O/'rehearsal_raw.json').read_text());assert raw['complete'] and not raw['failures']
assert raw['task_checks']==2*12*(2+len(meta['arms'])) and raw['nearby_task_checks']==2*2*len(meta['nearby_arms'])
wire=lab_packet.encode_packet(p.stem,meta['payload_sha256'][:16],dict(lines=lines));rec=lab_packet.read_packets(wire);assert len(rec)==1
encoded=next(v[len('P3A22_XZ85 '):] for v in lines if v.startswith('P3A22_XZ85 '));envelope=json.loads(lzma.decompress(base64.b85decode(encoded)));decoded=p3a_unpack_vectors(codec.s21_unpack_evidence(envelope))
def compact(x):
 if isinstance(x,float):return round(x,6)
 if isinstance(x,list):return [compact(v) for v in x]
 if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
 return x
assert decoded==compact(raw)
check=codec.s21_decode_samples(copy.deepcopy(decoded));negative=promotion.evaluate(dict(id='LOCAL-SCREEN-ONLY',result=dict(passed=False,gpu='sm120')),check,meta);assert not negative['measured'] and not negative['freeze_gate']
# Synthetic high-entropy64-seed/16-round capacity test, not candidate timing.
stress=copy.deepcopy(decoded);stress=codec.s21_decode_samples(stress);rng=random.Random(20261027)
for label,cell in stress['loads'].items():
 cell['n']=64
 def value(name):
  if name=='A':return 1.
  radius=15000 if name=='A2' or label in ('verify 16','verify 32') else 100000
  return 1+rng.randrange(-radius,radius+1)/1e6
 cell['samples']=[{name:value(name) for name in stress['sample_arms']} for _ in range(64)]
stress.update(seeds=64,seed_ids=meta['confirmation_seeds'],measurement_protocol=meta['measurement_protocol'])
for cell in stress['timing_round_counts'].values():
 for name in cell:cell[name]=[16]*64
for status in stress['arm_status'].values():status['native']=False;status['seed_ids']=meta['confirmation_seeds']
for trace in stress['timing_diagnostics'].get('traces',[]):
 if isinstance(trace,dict):
  index=rng.randrange(64);trace['seed_index']=index;trace['seed']=meta['confirmation_seeds'][index];trace['orders']=p3a_orders(stress['sample_arms'],16,index)
  base=trace['telemetry'][0];trace['telemetry']=[dict(base,t_s=round(base['t_s']+j*.013,3),power_mW=rng.randrange(150000,600001),temperature_C=rng.randrange(33,61)) for j in range(9)]
  for field in ('round_us','host_enqueue_us'):
   if field in trace:
    for name in trace[field]:trace[field][name]=[round(struct.unpack('<f',struct.pack('<f',32*rng.randrange(4000,14000)/1e6))[0]*1000,6) if field=='round_us' else rng.randrange(18000,40000)/1000 for _ in range(16)]
stress.update(codec.s21_pack_samples(stress['loads'],stress['sample_arms'],1000000))
for cell in stress['loads'].values():cell.pop('samples')
assert p3a_unpack_vectors(p3a_pack_vectors(compact(stress)))==compact(stress)
big=base64.b85encode(lzma.compress(json.dumps(codec.s21_pack_evidence(p3a_pack_vectors(compact(stress))),separators=(',',':')).encode())).decode()
stress_wire=lab_packet.encode_packet(p.stem,meta['payload_sha256'][:16],dict(lines=['META SYNTHETIC TRANSPORT ONLY','P3A22_XZ85 '+big,'END complete=False']))
assert len(stress_wire.encode())+1<=11744,len(stress_wire.encode())+1
assert len(lab_packet.read_packets((stress_wire+'\n'+'P'*4000)[-16000:]))==1
out=dict(payload_sha256=meta['payload_sha256'],wrapper_sha256=meta['wrapper_sha256'],base_sha256=meta['sha256'],complete=True,local_seed_count=2,local_rounds=2,TASK=raw['task_checks'],nearby=raw['nearby_task_checks'],all12shapes=True,all_arms=list(meta['arms']),native_H200=False,local_hooks='S22 forced stand-ins,CACHE1 activation-bank global stand-in',exact_archive_round_trip=True,local_cannot_promote=True,local_wire_bytes=len(wire.encode())+1,synthetic64x16_transport_bytes=len(stress_wire.encode())+1,synthetic_is_not_a_timing_measurement=True,tail4000_pass=True,seconds=time.monotonic()-t)
(D/'rehearsal_validation.json').write_text(json.dumps(out,indent=2)+'\n');print('PACKED_REHEARSAL_PASS',json.dumps(out),flush=True)
