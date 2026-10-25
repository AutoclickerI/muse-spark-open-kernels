"""Aggregate exact final source IDs, task coverage, decoder and native compile proof."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
arms=['w1k8','w1k8sp','w2k8','w2k8sp','w1k4raw','w1k4sp','w1k4s4','w1k4s4sp','w1k8spvec','w2k8spvec','w1k4spvec','w1k4s4spvec']
sha=hashlib.sha256((H/'candidate.py').read_bytes()).hexdigest();rows=[]
for name in arms:
 d=json.loads((H/'screen'/name/'result.json').read_text());assert d['sha256']==sha
 checks={k:dict(count=len(d[k]),passed=sum(x['ok'] for x in d[k])) for k in ('tests','shapes','static')}
 assert checks['tests']['count']==9 and checks['shapes']['count']==30
 assert all(x['count']==x['passed'] for x in checks.values()),(name,checks)
 rows.append(dict(arm=name,sha256=sha,checks=checks,prepare_s=d['prepare_s']))
native=json.loads((H/'native_compile.json').read_text());assert len(native)==12 and all(r['ok'] for r in native)
assert all(all(tuple(map(int,v))==(0,0,0) for v in r['stack_spills']) for r in native)
bits=json.loads((H/'bit_validation.json').read_text());assert bits['passed'] and len(bits['rows'])==6
lab=json.loads((H/'lab_validation.json').read_text());assert lab['passed'] and lab['rows']==48
r=dict(candidate_sha256=sha,payload_sha256=hashlib.sha256((H/'p10_sparse_pool_lab.py').read_bytes()).hexdigest(),payload_bytes=(H/'p10_sparse_pool_lab.py').stat().st_size,host_sha256=hashlib.sha256((H/'host_lab.py').read_bytes()).hexdigest(),host_bytes=(H/'host_lab.py').stat().st_size,screens=rows,native_compile_units=12,native_zero_spills=True,bit_roundtrips=6,local_lab=dict(comparisons=48,seeds=8,rounds=12,hopper_cluster='explicit arithmetic launch stand-in'))
(H/'admission.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS',r['candidate_sha256'][:8],r['payload_sha256'][:8],len(rows),'arms;12native;48lab rows')
