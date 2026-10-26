"""Reconstruct the S20 tail pair and audit the three previous native A:A archives."""
import ast,base64,hashlib,json,lzma,math,statistics,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import paired_evidence
from promotion import interval
from h200_lab import lab_payloads
JOBS=[('0640cce4','S20'),('6ac9317e','S19'),('7c0fd3cc','S19'),('5d0282bb','S18')]
def decode(raw):
    names=raw['sample_arms'];binary=base64.b85decode(raw.pop('sample_i32_b85'));values=iter(struct.unpack('<'+'i'*(len(binary)//4),binary))
    for label in raw['sample_loads']:
        raw['loads'][label]['samples']=[{n:1+next(values)/raw['sample_scale'] for n in names} for _ in range(raw['loads'][label]['n'])]
    return raw

def archived_timing_source(path):
    pending=[x[2] for x in lab_payloads(path)];seen=set()
    while pending:
        source=pending.pop()
        if source in seen:continue
        seen.add(source)
        try:tree=ast.parse(source)
        except SyntaxError:continue
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ab_time']
        if functions:
            fn=ast.get_source_segment(source,functions[-1])
            assert 'names[r % len(names):]' in fn and 'order[::-1]' in fn
            calls=[ast.unparse(n.args[1]) for n in ast.walk(tree) if isinstance(n,ast.Call) and getattr(n.func,'id','')=='ab_time']
            return dict(sha256=hashlib.sha256(fn.encode()).hexdigest(),function=fn,rounds_expressions=calls)
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode':
                try:pending.append(lzma.decompress(base64.b85decode(ast.literal_eval(n.args[0]))).decode())
                except (ValueError,TypeError,lzma.LZMAError,UnicodeDecodeError):pass
    raise AssertionError('No archived timing function in '+str(path))

def main():
    archives=[]
    for prefix,kind in JOBS:
        p=next((R/'runs/muse_prefill_layer').glob('*'+prefix+'*.json'));d=json.loads(p.read_text());raw=decode(paired_evidence(d['result']['log_tail'],kind))
        timing_source=archived_timing_source(p.with_suffix('.py'))
        rows={label:v['samples'] for label,v in raw['loads'].items()}
        aa=[math.exp(statistics.mean(math.log(v[i]['A2']/v[i]['A']) for v in rows.values())) for i in range(raw['seeds'])]
        load_aa={label:interval([r['A2']/r['A'] for r in v]) for label,v in rows.items()}
        primary=[(r['INC']/r['A'],label,i,raw['seed_ids'][i]) for label,v in rows.items() for i,r in enumerate(v)]
        load16=[r['INC']/r['A'] for r in rows['verify 16']]
        exposure={}
        for label,shape in raw['shapes'].items():
            names=['A']+list(shape['r']);positions={n:[] for n in names}
            # All four archived helpers use this exact rotate/reverse loop.
            rounds=16 if label in ('d1S','d1F','v1S','v1F') else 8
            for rnd in range(rounds):
                order=names[rnd%len(names):]+names[:rnd%len(names)]
                if rnd%2:order=order[::-1]
                for slot,name in enumerate(order):positions[name].append(rnd*len(names)+slot)
            means={n:statistics.mean(v) for n,v in positions.items()}
            exposure[label]=dict(arms=len(names),rounds=rounds,mean_slot=means,AA_slot_difference=means['A2']-means['A'])
        archives.append(dict(job=d['id'],receipt=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),machine=d['result']['machine'],
            timing_source=timing_source,native=raw.get('arm_status',{}).get('A',{}).get('native',d['result']['gpu']=='H200'),seed_count=raw['seeds'],AA_G0=interval(aa),AA_loads=load_aa,
            max_primary=dict(zip(('ratio','load','seed_index','seed'),max(primary))),verify16_range=[min(load16),max(load16)],
            verify16_AA_range=[min(r['A2']/r['A'] for r in rows['verify 16']),max(r['A2']/r['A'] for r in rows['verify 16'])],
            v16S_summary=raw['shapes']['v16S'],order_exposure=exposure,round_data_present=any(k in raw for k in ('rounds','round_times','timing_rounds','round_archive'))))
        if prefix=='0640cce4':
            i=raw['seed_ids'].index(1730886002);row=rows['verify 16'][i]
            culprit=dict(arm='INC (S19 62ea5bf4)',shape='v16S inferred from shape aggregate; per-seed S/F split not archived',load='verify 16',seed=1730886002,seed_index=i,
                load_ratios=row,round=None,round_reason='Only arithmetic means reached per-seed load archive; eight per-arm replay times were discarded by producer',
                rounds_in_source=8,timing_order='rotate(names,r); reverse on odd r',telemetry_scope='run endpoints only; no clock/temperature/power at the pair',
                transient_assessment='Isolated INC-only latency anomaly is consistent with a transient, but not proven judge-side; no per-round trace or simultaneous device telemetry')
    covered=sum(a['AA_G0']['CI95'][0]<=1<=a['AA_G0']['CI95'][1] for a in archives)
    out=dict(culprit=culprit,archives=archives,AA_coverage=dict(joint_global_intervals=len(archives),contain_unity=covered,
        caveat='Four run-level archives are insufficient to validate a new robust confidence estimator; per-load intervals are correlated'),
        decision='Keep 0640cce4 failed and all samples. Retain Student-t held-out seed gate in tools/promotion.py. Prospectively use 64 fixed fresh seeds and symmetric palindromic cohort order (ABBA for two arms), retain round diagnostics and NVML snapshots. No selective reruns, trimming or optional stopping.')
    (H/'diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(culprit=culprit,previous=[dict(job=a['job'],verify16_range=a['verify16_range'],AA_G0=a['AA_G0']) for a in archives],AA_coverage=out['AA_coverage']),indent=2))
if __name__=='__main__':main()
