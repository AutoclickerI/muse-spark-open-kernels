"""S16 whole-layer admission; freeze only on complete per-piece paired wins and controls."""
import argparse,base64,hashlib,json,lzma,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
    receipt=json.loads(a.receipt.read_text());manifest=json.loads((H/'generated/lab_manifest.json').read_text())
    assert hashlib.sha256(a.receipt.with_suffix('.py').read_bytes()).hexdigest()==manifest['wrapper_sha256']
    log=receipt['result']['log_tail'];blob=next(line.split(' ',1)[1] for line in log.splitlines() if line.startswith('S16_XZ85 '))
    raw=json.loads(lzma.decompress(base64.b85decode(blob)));assert raw['host_sha256']==manifest['sha256']
    wins={};controls={}
    def rows(arm):return {tuple(r) for r in manifest['arms'][arm]['rows']}
    for label,shape in raw['shapes'].items():
        key=(int(re.search(r'\d+',label)[0]),1 if label.startswith('d') else 8,label.endswith('S'))
        radius=max(abs(v-1) for v in shape['aa'])
        for arm,value in shape['r'].items():
            if arm!='A2' and key not in rows(arm):controls[arm+':'+label]=abs(value-1)<=radius
    for load,cell in raw['loads'].items():
        kind,b=load.split();b=int(b);q=1 if kind=='decode' else 8;radius=max(abs(v-1) for v in cell['aa'])
        for arm,value in cell['r'].items():
            if arm=='A2':continue
            if any((b,q,w) in rows(arm) for w in (True,False)):
                wins[arm+':'+load]=dict(ratio=value['mean'],AA_radius=radius,wins=value['mean']>1+radius)
            else:controls[arm+':'+load]=abs(value['mean']-1)<=radius
    gate=bool(receipt['result'].get('passed') and raw['complete'] and all(v['wins'] for v in wins.values()) and all(controls.values()))
    result=dict(job=receipt['id'],source_sha256=manifest['sha256'],freeze_gate=gate,wins=wins,controls=controls,raw=raw)
    (H/'s16_report.json').write_text(json.dumps(result,indent=2)+'\n')
    if gate:
        source=(R/manifest['candidate']).read_text();assert hashlib.sha256(source.encode()).hexdigest()==manifest['sha256']
        from pack import pack
        final=R/'problems/muse_decode_layer/cand_r3_best7_s16_final.py';final.write_text(pack(source))
        print(final,hashlib.sha256(final.read_bytes()).hexdigest(),'expanded',manifest['sha256'])
    else:print('S16 not frozen; retain report and remove/retest losing or noisy pieces')


if __name__=='__main__':main()
