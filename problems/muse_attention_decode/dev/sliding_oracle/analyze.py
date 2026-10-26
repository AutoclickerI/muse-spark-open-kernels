"""Paired oracle removal costs; keep altered work separate from candidate ratios."""
import argparse,base64,json,lzma,math,statistics
from pathlib import Path
from scipy.stats import t
H=Path(__file__).resolve().parent

def stats(values):
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values));q=float(t.ppf(.975,len(values)-1))
    return dict(mean=mean,SE=se,CI95=[mean-q*se,mean+q*se],n=len(values))

def unpack(path):
    receipt=json.loads(path.read_text());blob=next(s.split(' ',1)[1] for s in receipt['result']['log_tail'].splitlines() if s.startswith('SLIDING_ORACLE_XZ85 '))
    return receipt,json.loads(lzma.decompress(base64.b85decode(blob)))

def analyze(path):
    receipt,d=unpack(path);rows=[]
    pairs=[('coordinates','M','C'),('history_and_new_softmax_pack','C','S'),('merge_after_constant_softmax','S','X'),('QK_PV_transpose_newKV','X','L'),('KV_transport_without_math','L','Z'),('remaining_query_setup_address_copy_issue','Z','E'),('launch_cluster_schedule','E','G')]
    for r in d['results']:
        n=len(r['seeds']);assert all(v==n for v in r['TASK'].values());v=r['times_ns']
        ratios={a+'/'+b:stats([x/y for x,y in zip(v[a],v[b])]) for a,b in [('O2','O'),('M2','M'),('M','O'),('C','M'),('S','C'),('N','M'),('X','S'),('W','M')]+([('O','F')] if 'F' in v else [])}
        split={name:stats([(a-b)/1000 for a,b in zip(v[left],v[right])]) for name,left,right in pairs}
        for i in range(n):assert abs(sum((v[a][i]-v[b][i])/1000 for _,a,b in pairs)-(v['M'][i]-v['G'][i])/1000)<1e-8
        profiles={}
        for arm,ns in r['profiles'].items():
            profiles[arm]=[dict(name=d['names'][q[0]],start_ns=q[1],end_ns=q[2],duration_us=(q[2]-q[1])/1000,grid=q[3:6],threads=q[6],shared=q[7],registers=q[8]) for q in ns]
        rows.append(dict(B=r['B'],L=r['L'],seed_ids=r['seeds'],ratios=ratios,ordered_split_us=split,
            net_stage_us={arm:stats([(x-y)/1000 for x,y in zip(xs,v['G'])]) for arm,xs in v.items() if arm!='G'},
            full_no_merge_removed_us=stats([(a-b)/1000 for a,b in zip(v['M'],v['N'])]),
            ready_wait_removed_us=stats([(a-b)/1000 for a,b in zip(v['M'],v['W'])]),resources=r['resources'],clocks=r['clocks'],profiles=profiles))
    result=dict(job=receipt['id'],receipt=str(path),rows=rows,source_sha256=d['source_sha256'],elapsed_s=d['elapsed_s'],
        limitations=['The telescoping split is order-dependent exposed removal cost, not independent additive hardware utilization.',
          'All M modes use one loaded binary. M/O reports changes caused by instrumentation branches/compiler allocation; same geometry does not imply identical code to O.',
          'Coordinates are precomputed 64-bit byte offsets with their own cold reads; C is an oracle, never a candidate.',
          'PDL protocol is measured on ready input. Active producer waiting is not measured by this stage oracle.',
          'Clocked source has different instructions/resources; phase cycles diagnose the path, uninstrumented seed timings rank it.',
          'E-G is launch/cluster scheduling and minimal instructions, not host dispatch; L-Z is transport exposed without tensor/transpose work.'])
    (H/'oracle_result.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(r['B'],r['L'],'ratios',json.dumps(r['ratios']))
        print('split_us',{k:round(v['mean'],4) for k,v in r['ordered_split_us'].items()})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();analyze(a.receipt)
