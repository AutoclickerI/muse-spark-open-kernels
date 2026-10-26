"""Read checked structured packets; keep oracles separate from valid candidates."""
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
from scipy.stats import t
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets
def stats(xs):
    m=statistics.mean(xs);se=statistics.stdev(xs)/math.sqrt(len(xs));q=float(t.ppf(.975,len(xs)-1))
    return dict(mean=m,SE=se,CI95=[m-q*se,m+q*se],n=len(xs))
def analyze(path,hoist=False):
    receipt=json.loads(path.read_text());records=read_packets(receipt['result'].get('log_tail',''));assert len(records)==1
    p=records[0]['packet'];assert p['complete'] and not p.get('transport_errors'),p.get('transport_errors')
    queue=json.loads((H/('hoist_queue.json' if hoist else 'queue.json')).read_text());assert records[0]['source_sha256']==queue['payload_sha256'][:16]
    expected={x['name'] if x['name'].endswith('M') else x['name']+'A':x['sha256'] for x in json.loads((H/('native_hoist.json' if hoist else 'native.json')).read_text())}
    assert p['identities']['sources']==expected
    rows=[];nominations=[]
    for r in p['paired']:
        n=len(r['seeds']);assert n==(16 if hoist else 12) and len(set(r['seeds']))==n and all(v==n for v in r['TASK'].values()) and r['oracle_TASK']==(0 if hoist else 2*n)
        v=r['times_ns'];ratios={a+'/'+b:stats([x/y for x,y in zip(v[a],v[b])]) for a,b in [('A2','A'),('RAW','A')]+([('H','A'),('H','RAW')] if hoist else [('Q','A'),('Q','RAW')]+([('T','A'),('T','RAW')] if r['L']==8 else []))}
        o=r['oracle_ns'];split={} if hoist else {name:stats([(x-y)/1000 for x,y in zip(o[a],o[b])]) for name,a,b in [('projection_value_loads','Z','ZV'),('rope_sincos','ZV','ZR'),('normalization_rotation_pack_newKV_gate','ZR','ZQ'),('zero_byte_copy_issue','ZQ','ZI'),('offset_reads_address_and_loop','ZI','ZL'),('remaining_protocol_schedule','ZL','E'),('launch','E','G')]}
        if not hoist:
            for i in range(n):assert abs(sum(x[i]-o[b][i] for x,b in [(o[a],b) for a,b in [('Z','ZV'),('ZV','ZR'),('ZR','ZQ'),('ZQ','ZI'),('ZI','ZL'),('ZL','E')]])-(o['Z'][i]-o['E'][i]))<1e-6
        point=[]
        for arm in (('H','RAW') if hoist else ('Q','T','RAW')):
            if arm not in v or arm=='RAW' and (r['B'],r['L'])!=(16,8):continue
            if ratios[arm+'/A']['mean']<1:
                nom=dict(B=r['B'],L=r['L'],arm=arm,control='S19',**ratios[arm+'/A'],A_A=ratios['A2/A'],seed_ids=r['seeds']);point.append(nom);nominations.append(nom)
        row=dict(B=r['B'],L=r['L'],seed_ids=r['seeds'],ratios=ratios,ordered_oracle_split_us=split,point_nominations=point,critical=r['critical'],resources=r['resources'])
        if not hoist:row.update(oracle_A_A=stats([x/y for x,y in zip(o['M2'],o['M'])]),oracle_prep_removed_us=stats([(x-y)/1000 for x,y in zip(o['Z'],o['ZQ'])]),oracle_residual_us=stats([(x-y)/1000 for x,y in zip(o['Z'],o['E'])]))
        rows.append(row);print(r['B'],r['L'],json.dumps(ratios));print('oracle_us',{k:round(s['mean'],4) for k,s in split.items()})
    out=dict(job=receipt['id'],receipt=str(path),base=p['identities']['base_sha256'],payload_sha256=queue['payload_sha256'],rows=rows,nominees=nominations,TASK=p['checks'],elapsed_s=p['elapsed_s'],raw=p,
             limitations=['Ordered removals, not independent hardware attribution.','Oracle input is ready BF16; actual S19 partials are consumed only in valid whole-layer arms.','A2/A and M2/M retained; no per-shape significance veto.','P4 confirms six-load G0 on held-out inputs.'])
    (H/('hoist_result.json' if hoist else 'result.json')).write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--hoist',action='store_true');a=ap.parse_args();analyze(a.receipt,a.hoist)
