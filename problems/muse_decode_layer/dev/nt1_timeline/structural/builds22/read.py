"""Whole-load S22 evidence through promotion.py; no component-time score transfer."""
import argparse,ast,base64,hashlib,importlib.util,json,lzma,math,sys
from pathlib import Path
from sample_codec import p3a_unpack_vectors
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import lab_packet,promotion
sp=importlib.util.spec_from_file_location('p3a22_codec_read',R/'problems/muse_decode_layer/dev/s21/evidence.py');codec=importlib.util.module_from_spec(sp);sp.loader.exec_module(codec)
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def analyze(path,manifest_path=None):
 path=Path(path);m=json.loads(Path(manifest_path or D/'lab_manifest.json').read_text());host=json.loads(path.read_text());wrapper=path.with_suffix('.py').read_text();assert sha(wrapper)==m['wrapper_sha256'] and host['result']['gpu']=='H200'
 entries=next(ast.literal_eval(n.value) for n in ast.parse(wrapper).body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='_LAB_PAYLOADS' for t in n.targets));name,short,payload=next(v for v in entries if v[1]==m['payload_sha256'][:16]);assert sha(payload)==m['payload_sha256']
 blob=next(n.args[0].value for n in ast.walk(ast.parse(payload)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');body=lzma.decompress(base64.b85decode(blob)).decode();assert sha(body)==m['expanded_payload_sha256']
 bnode=next(n for n in ast.parse(body).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='S16_BUNDLE');blob=next(n.args[0].value for n in ast.walk(bnode.value) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');bundle=json.loads(lzma.decompress(base64.b85decode(blob)));assert sha(bundle['host'])==m['sha256']
 for arm,spec in bundle['arms'].items():
  lines=bundle['host'].splitlines(True)
  for a,b,t in reversed(spec['edits']):lines[a:b]=[t]
  assert sha(''.join(lines))==spec['sha256']==m['arms'][arm]['sha256']
 evidence=lab_packet.paired_evidence(host['result']['log_tail'],'P3A22')
 if evidence is None:
  out=dict(job=host['id'],measured=False,reason='No complete checksummed P3A22 archive',rows=[]);(D/'results.json').write_text(json.dumps(out,indent=2)+'\n');return out
 raw=codec.s21_decode_samples(p3a_unpack_vectors(codec.s21_unpack_evidence(evidence)));full=promotion.evaluate(dict(id=host['id'],result=dict(passed=False,gpu='H200')),raw,m);outcomes={};rows=[]
 for name in m['active']:
  f=full['pieces'][name];out=dict(measured=f['measured'],source_sha256=m['arms'][name]['sha256'],whole_graph=True)
  if f['measured']:
   rr=[1/x for x in f['per_seed_R']];aa=[math.exp(sum(math.log(v['samples'][i]['A2']) for v in raw['loads'].values())/6) for i in range(len(rr))];out.update(promotion.interval(rr,aa));out['per_seed_R']=rr
   status=[full['arm_status'][x] for x in ('A','A2',name)];nearby=all(x['nearby_task_checks']==x['nearby_expected_checks'] and not x['failures'] for x in status)
   out['lab_gate']=out['CI95'][0]>1 and nearby;out['TASK_nearby_pass']=nearby;out['official_TASK_pending']=True;out['freeze_gate']=False
   out['loads']={load:promotion.interval([1/v[name] for v in cell['samples']],[v['A2'] for v in cell['samples']]) for load,cell in raw['loads'].items()}
   out['prepare_upper_s']=raw['host_prepare_s']+raw['p3a_prepare_delta_s'][name];out['prepare_note']='Shared host prepare+arm delta is a conservative bound,not a separate cold prepare measurement'
   points=[]
   for label,cell in raw['p3a_shape_moments_nano'].items():
    mean,se=cell[name];mean/=1e9;se/=1e9;aa_se=cell['A2'][1]/1e9;err=promotion.T95[min(len(rr)-1,30)]*max(se,aa_se);M=int(label[1:-1])*(8 if label.startswith('v') else 1)
    points.append(dict(label=label,M=M,time_ratio=math.exp(mean),time_CI95=[math.exp(mean-err),math.exp(mean+err)],log_SE=se,AA_log_SE=aa_se,seeds=len(rr),control_alias=M>32,point_nominee=M<=32 and mean<0))
   out['shapes']=points
   for point in points:
    if not point['point_nominee']:continue
    label=point['label'];B=int(label[1:-1]);Q=8 if label[0]=='v' else 1
    rows.append(dict(arm=name,job=host['id'],batch='P3a S22 production graph',M=point['M'],shapes=[[B,Q,label[-1]=='S']],ratio=point['time_ratio'],SE=point['time_ratio']*point['log_SE'],seeds=len(rr),seed_ids=raw['seed_ids'],CI95=point['time_CI95'],AA_log_SE=point['AA_log_SE'],whole_six_load_R=out['R'],whole_six_load_CI95=out['CI95'],whole_lab_gate=out['lab_gate'],exact_patch=m['candidate_sources'][name],handoff=True,scope='Whole-layer shape point nomination;P4 composes and confirms on new seeds;M16/M32 share a native source specialization'))
  outcomes[name]=out
 out=dict(job=host['id'],measured=all(v['measured'] for v in outcomes.values()),source_audit=dict(verified=True,wrapper_sha256=m['wrapper_sha256'],payload_sha256=m['payload_sha256'],base_sha256=m['sha256'],candidate_sha256={n:a['sha256'] for n,a in m['arms'].items()}),protocol=m['measurement_protocol'],outcomes=outcomes,rows=rows,AA=full['score'],checks={k:raw[k] for k in ('complete','failures','task_checks','expected_task_checks','nearby_task_checks','nearby_expected_checks')},base_prepare_s=raw['host_prepare_s'],raw=raw,official_freeze=False)
 (D/'results.json').write_text(json.dumps(out,indent=2)+'\n');(D/'P4_HANDOFF.json').write_text(json.dumps(dict(base='S22 '+m['sha256'],job=host['id'],rows=rows,outcomes=outcomes,source_audit=out['source_audit'],gate='Whole production graph measurements;any valid per-shape point nominee retained. P4 new held-out six-load R lower bound>1 and official TASK finish promotion'),indent=2)+'\n')
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--manifest',type=Path);a=ap.parse_args();out=analyze(a.receipt,a.manifest)
 print(json.dumps({k:v for k,v in out.items() if k not in ('raw','rows')},indent=2))
