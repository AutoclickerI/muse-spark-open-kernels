"""S19 vs frozen S18 from a lab receipt: R and CI by tools/promotion.py, per-arm diagnostics, prepare seconds."""
import argparse,base64,hashlib,json,lzma,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from promotion import evaluate
from lab_packet import result_lines, paired_evidence


def decode(raw):
    names=raw['sample_arms'];labels=raw['sample_loads'];scale=raw['sample_scale']
    binary=base64.b85decode(raw.pop('sample_i32_b85'));count=len(names)*sum(raw['loads'][label]['n'] for label in labels)
    assert len(binary)==4*count;values=struct.unpack('<'+'i'*count,binary);offset=0
    for label in labels:
        rows=[]
        for _ in range(raw['loads'][label]['n']):rows.append({name:1+values[offset+j]/scale for j,name in enumerate(names)});offset+=len(names)
        raw['loads'][label]['samples']=rows
    assert offset==count;return raw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--official-task',type=Path,help='candidate decode TASK receipt, required for promotion');a=ap.parse_args()
    m=json.loads((H/'manifest.json').read_text());receipt=json.loads(a.receipt.read_text())
    assert hashlib.sha256(a.receipt.with_suffix('.py').read_bytes()).hexdigest()==m['wrapper_sha256']
    batch=H/'batches'/m['wrapper_sha256'][:16];batch.mkdir(parents=True,exist_ok=True)
    tail=receipt['result']['log_tail'];decoded_lines=result_lines(tail);evidence=paired_evidence(tail,'S19')
    if evidence is None:
        prep=[json.loads(l.split(' ',1)[1]) for l in decoded_lines if l.startswith('S19_PREPARE ')]
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
        result.update(gate=result['freeze_gate'],official_TASK_receipt=str(official) if official else None,
                      prepare={k:raw.get(k) for k in ('host_prepare_s','prepare_lines','build_s','delta_s')},
                      attention_plan=raw.get('attention_plan'),lab_end=[l for l in tail.splitlines() if l.startswith('LAB_END')])
    (batch/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw','pieces','score')},indent=2))
    if 'score' in result:
        s=result['score'];print('S19/S18 R',s['R'],s['CI95'],'promote',s['promote'])
        for k,v in s.get('loads',{}).items():print(' load',k,round(v['R'],5),[round(x,5) for x in v['CI95']])
        for arm,p in result['pieces'].items():print(' arm',arm,p['R'],p['CI95'],{k:round(v['R'],4) for k,v in p.get('loads',{}).items()})
if __name__=='__main__':main()
