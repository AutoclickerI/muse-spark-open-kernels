"""S18 held-out promotion against frozen S17, with lossless sample decoding."""
import argparse,base64,hashlib,json,lzma,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from promotion import evaluate
from lab_packet import result_lines, paired_evidence


def decode(raw):
    names=raw['sample_arms'];labels=raw['sample_loads'];scale=raw['sample_scale']
    binary=base64.b85decode(raw.pop('sample_i32_b85'));count=len(names)*sum(raw['loads'][label]['n'] for label in labels)
    assert len(binary)==4*count
    values=struct.unpack('<'+'i'*count,binary);offset=0
    for label in labels:
        rows=[]
        for _ in range(raw['loads'][label]['n']):
            rows.append({name:1+values[offset+j]/scale for j,name in enumerate(names)});offset+=len(names)
        raw['loads'][label]['samples']=rows
    assert offset==count
    return raw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
    receipt=json.loads(a.receipt.read_text());wrapper=hashlib.sha256(a.receipt.with_suffix('.py').read_bytes()).hexdigest()
    batch=H/'batches'/wrapper[:16];m=json.loads((batch/'lab_manifest.json').read_text());assert m['wrapper_sha256']==wrapper
    evidence=paired_evidence(receipt['result']['log_tail'],'S18')
    if evidence is None:
        errors=[l for l in receipt['result']['log_tail'].splitlines() if l.startswith(('LAB_ERROR','LAB_TRUNCATED'))]
        result=dict(job=receipt['id'],source_sha256=m['sha256'],freeze_gate=False,R=None,CI95=None,measured=False,
            host_passed=receipt['result'].get('passed'),reason='No paired seed archive; not a timing loss',errors=errors)
        (batch/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return
    result=evaluate(receipt,decode(evidence),m)
    (H/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (batch/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['freeze_gate']:
        p=R/m['shipped_source'];assert hashlib.sha256(p.read_bytes()).hexdigest()==m['shipped_sha256']
        (R/'problems/muse_decode_layer/cand_r3_best7_s18_final.py').write_bytes(p.read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw','pieces')},indent=2))
if __name__=='__main__':main()
