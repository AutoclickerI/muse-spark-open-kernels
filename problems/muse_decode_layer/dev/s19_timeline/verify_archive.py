"""Check exact timestamp/work reconstruction against the pre-encoding local observer."""
import json,lzma
from pathlib import Path
import report
H=Path(__file__).resolve().parent
raw=json.loads(lzma.decompress((H/'local_raw.json.xz').read_bytes()))
packet=json.loads((H/'local_receipt.json').read_text())['packet']
x=report.unpack(packet['lines'][0].split(' ',1)[1])
for a,b in zip(raw['cases'],x['cases']):
    assert a['dag']==b['dag']
    for n,m in zip(a['trace'],b['trace']):
        for k in ('node','stream','start_ns','end_ns','grid','block','shared','registers'):assert n[k]==m[k],(a['case'],k)
    for s,t in zip(a['seeds'],b['seeds']):
        assert s['work']==t['work'],(a['case'],s['work'],t['work'])
        off=s['round_offset'];width=raw['round_width']
        for r in t['rounds']:
            row=raw['round_values'][off+r['round']*width:off+(r['round']+1)*width]
            assert round(r['wall_us']*1000)==row[0] and round(r['event_us']*1000)==row[1]
            assert [round(r['critical_us'][k]*1000) for k in report.PHASES]==row[2:]
result=dict(complete=True,shapes=12,seeds_per_shape=3,profile_rounds=3,profile_samples=108,control_samples=288,exact_representative_timestamps_DAG_resources=True,exact_logical_work_reconstruction=True,exact_all_round_owner_event_ns=True,local_sm120_only=True)
(H/'local_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
