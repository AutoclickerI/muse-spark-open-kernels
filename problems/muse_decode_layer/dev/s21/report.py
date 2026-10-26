"""S21 vs S19: native R and leave-outs by tools/promotion.py; official TASK remains a separate gate."""
import argparse,base64,hashlib,json,lzma,math,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from promotion import evaluate,interval
from lab_packet import result_lines, paired_evidence


from evidence import s21_decode_samples,s21_unpack_evidence

def decode(raw):return s21_decode_samples(s21_unpack_evidence(raw))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--official-task',type=Path,help='candidate decode TASK receipt, required for promotion');a=ap.parse_args()
    m=json.loads((H/'manifest.json').read_text());receipt=json.loads(a.receipt.read_text())
    assert hashlib.sha256(a.receipt.with_suffix('.py').read_bytes()).hexdigest()==m['wrapper_sha256']
    batch=H/'batches'/m['wrapper_sha256'][:16];batch.mkdir(parents=True,exist_ok=True)
    tail=receipt['result']['log_tail'];decoded_lines=result_lines(tail);evidence=paired_evidence(tail,'S21')
    if evidence is None:
        prep=[json.loads(l.split(' ',1)[1]) for l in decoded_lines if l.startswith('S21_PREPARE ')]
        result=dict(job=receipt['id'],measured=False,reason='No paired seed archive; not a timing loss',prepare_failure=prep[0] if prep else None,
                    errors=[l for l in tail.splitlines() if l.startswith(('LAB_ERROR','LAB_TRUNCATED','LAB_END'))])
    else:
        raw=decode(evidence)
        official_pass=False
        official=a.official_task
        if official is None and m.get('official_TASK_run'):
            hits=list(R.glob(f"runs/muse_decode_layer/*_{m['official_TASK_run']}.json"))
            if len(hits)==1:official=hits[0]
        if official is not None:
            assert hashlib.sha256(official.with_suffix('.py').read_bytes()).hexdigest()==m['shipped_sha256']
            official_pass=json.loads(official.read_text())['result'].get('passed') is True
        result=evaluate(dict(id=receipt['id'],result=dict(passed=official_pass,gpu=receipt['result'].get('gpu'))),raw,m)
        if result['score']['measured']:
            aa=[];alternate=[]
            for i in range(len(raw['seed_ids'])):
                control=math.exp(sum(math.log(v['samples'][i]['A2']) for v in raw['loads'].values())/6)
                primary=math.exp(sum(math.log(v['samples'][i]['INC']) for v in raw['loads'].values())/6)
                aa.append(control);alternate.append(primary/control)
            result['control_diagnostics']=dict(AA=interval(aa),INC_over_A2=interval(alternate),
                scope='Diagnostic only; no primary subtraction, gate change or selective retiming')
        primary=[result['arm_status'][n] for n in ('A','A2','INC')]
        lab_gate=(result['score']['measured'] and result['score']['CI95'][0]>1 and all(
            not s['failures'] and s['nearby_task_checks']==s['nearby_expected_checks'] for s in primary))
        prepare_gate=0<raw.get('host_prepare_s',float('inf'))<=300
        result.update(gate=result['freeze_gate'],lab_gate=lab_gate and prepare_gate,prepare_gate=prepare_gate,official_TASK_receipt=str(official) if official else None,
                      prepare={k:raw.get(k) for k in ('host_prepare_s','prepare_lines','build_s','delta_s')},
                      measurement_protocol=raw.get('measurement_protocol'),timing_diagnostics=raw.get('timing_diagnostics'),
                      original_pair_repeat=raw.get('original_pair_repeat'),
                      attention_plan=raw.get('attention_plan'),lab_end=[l for l in tail.splitlines() if l.startswith('LAB_END')])
    (batch/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    # Canonical receipt retains every paired sample. Keep one compact review
    # result in git; the full derived archive in batches/ is rebuildable.
    slim=lambda d:{k:v for k,v in d.items() if k not in ('seed_ids','selection_seeds','per_seed_R')}
    summary={k:v for k,v in result.items() if k not in ('raw','arm_status','pieces','score')}
    if 'score' in result:
        summary.update(score=slim(result['score']),pieces={k:slim(v) for k,v in result['pieces'].items()},
            checks={k:result['raw'][k] for k in ('complete','failures','task_checks','expected_task_checks','nearby_task_checks','nearby_expected_checks')})
    (H/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw','pieces','score','arm_status','timing_diagnostics','original_pair_repeat')},indent=2))
    if 'score' in result:
        s=result['score'];print('S21/S19 R',s['R'],s['CI95'],'promote',s['promote'])
        for k,v in s.get('loads',{}).items():print(' load',k,round(v['R'],5),[round(x,5) for x in v['CI95']])
        for arm,p in result['pieces'].items():print(' arm',arm,p['R'],p['CI95'],{k:round(v['R'],4) for k,v in p.get('loads',{}).items()})
if __name__=='__main__':main()
