"""Read retained timeline evidence without charging PDL residence twice."""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'problems/muse_decode_layer/dev/s15_timeline'))
from analysis import chain_account,model_roof
p=R/'problems/muse_decode_layer/dev/s15_timeline/out/h200/timeline.json';x=json.loads(p.read_text());rows={}
for cell in x['cases']:
    if not cell['case'].endswith('F'):continue
    account=chain_account(cell['trace'],cell['dag']);parts={v['node']:v for v in account['path']}
    seed=next(s for s in cell['seeds'] if s['seed']==cell['trace_seed']);work=seed['work']['attention_full'];roof=model_roof(work)
    kernels=[]
    for t in cell['trace']:
        if t['role']!='attention_full':continue
        k=dict(name=t['name'],grid=t['grid'],block=t['block'],shared=t['shared'],registers=t['registers'],
            residence_us=(t['end_ns']-t['start_ns'])/1000,critical_us=parts.get(t['node'],{}).get('critical_us',0),gap_us=parts.get(t['node'],{}).get('gap_us',0))
        kernels.append(k)
    critical=sum(k['critical_us'] for k in kernels);main=max(kernels,key=lambda k:k['residence_us'])
    rows[cell['case']]=dict(seed=seed['seed'],logical_read_bytes=work['read_bytes'],roof_us=roof['roof_us'],critical_us=critical,
        above_us=max(0,critical-roof['roof_us']),main_logical_TBs=work['read_bytes']/main['residence_us']/1e6,kernels=kernels)
out=dict(receipt='7eaa262a30ec483e',timeline_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='retained median-seed real S15 whole-layer trace; logical bandwidth, not hardware DRAM counters',
    stream_roof_TBs=4.58,rows=rows)
(H/'roof/timeline_account.json').write_text(json.dumps(out,indent=2)+'\n')
for name,r in rows.items():print(name,'raw critical/roof',r['critical_us'],r['roof_us'],'logical mainTB/s',r['main_logical_TBs'],[(k['name'],k['critical_us']) for k in r['kernels']])
