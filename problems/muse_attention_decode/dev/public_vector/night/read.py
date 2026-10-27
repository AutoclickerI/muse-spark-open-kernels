"""Whole-production-graph G0 points from the unchanged P4 sampling protocol."""
import argparse,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4];sys.path.insert(0,str(R/'tools'))
from lab_packet import paired_evidence
from promotion import evaluate,interval
sp=importlib.util.spec_from_file_location('p5b_night_codec',R/'problems/muse_decode_layer/dev/s21/evidence.py');codec=importlib.util.module_from_spec(sp);sp.loader.exec_module(codec)
def read(path):
 receipt=json.loads(path.read_text());sent=hashlib.sha256(path.with_suffix('.py').read_bytes()).hexdigest()
 m=next(m for m in map(lambda p:json.loads(p.read_text()),H.glob('lab_manifest_*.json')) if m['wrapper_sha256']==sent)
 tail=receipt['result'].get('log_tail','');e=paired_evidence(tail,'P5N')
 if e is None:return dict(job=receipt['id'],lab=m['name'],measured=False,reason='No complete paired evidence',status=receipt['result'].get('status'),
                          progress=[l for l in tail.splitlines() if l.startswith(('P5N_STEP','P5N_STALL','P5N_KILL'))][-40:])
 raw=codec.s21_decode_samples(codec.s21_unpack_evidence(e));full=evaluate(receipt,raw,m)
 rows={};aa=[math.exp(statistics.mean(math.log(v['samples'][i]['A2']) for v in raw['loads'].values())) for i in range(len(raw['seed_ids']))]
 for name,spec in m['arms'].items():
  health=full['pieces'][name]
  row=dict(measured=health['measured'],logical=spec['logical'],source_sha256=spec['sha256'])
  if row['measured']:
   values=[math.exp(statistics.mean(-math.log(v['samples'][i][name]) for v in raw['loads'].values())) for i in range(len(raw['seed_ids']))]
   row.update(interval(values,aa));row['loads']={label:interval([1/s[name] for s in v['samples']],[s['A2'] for s in v['samples']]) for label,v in raw['loads'].items()}
   row['nominate']=row['R']>1 or any(v['R']>1 for v in row['loads'].values())
   row['lab_G0']=row['CI95'][0]>1
  rows[name]=row
 out=dict(job=receipt['id'],lab=m['name'],progress=raw.get('p5n_progress'),measured=all(r['measured'] for r in rows.values()),comparison='S22/candidate whole load;46S+16F;larger is faster',AA=interval(aa),arms=rows,seeds=raw['seed_ids'],checks={k:raw[k] for k in ('complete','failures','task_checks','expected_task_checks','nearby_task_checks','nearby_expected_checks')},stage_resources=raw.get('stage_resources'),protocol=raw['measurement_protocol'],promotion='P4 freshly confirms any selected composition on disjoint seeds; no official score sent by this lane',prepare_caveat='Lab base prepare includes all-arm independent compile pool; candidate-specific300s gate needs P4 confirmation.')
 (H/('results_'+m['name']+'.json')).write_text(json.dumps(out,indent=2)+'\n');(H/'out'/(m['name']+'_receipt_data.json')).write_text(json.dumps(raw)+'\n');return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();print(json.dumps(read(a.receipt),indent=2))
