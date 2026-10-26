"""Read a full bounded receipt and audit its exact submitted payload/cells/selectors."""
import argparse,ast,base64,hashlib,json,lzma,subprocess,sys
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[4]
sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines,read_packets
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args();D=a.directory.resolve();receipt=a.receipt.resolve()
subprocess.run([sys.executable,str(S.parent/'review/read.py'),str(receipt)],check=True,stdout=subprocess.DEVNULL)
result=json.loads(receipt.with_suffix('.summary.json').read_text());queue=json.loads((D/'queue.json').read_text());payload=json.loads((D/'payload.json').read_text());source=(R/queue['source']).read_text();sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
assert sha(source)==queue['sha256'];nodes=ast.parse(source).body
emb=next(ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='_LAB_PAYLOADS' for t in n.targets));name,psha,body=next(e for e in emb if e[1]==payload['sha256'][:16]);assert sha(body)==payload['sha256']
blob=next(n.args[0].value for n in ast.walk(ast.parse(body)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');expanded=lzma.decompress(base64.b85decode(blob)).decode();assert sha(expanded)==payload['expanded_sha256']
cells=json.loads((D/'cells.json').read_text())
if D.name in ('dequant','occupancy','overlap'):cells={**{k:v for k,v in json.loads((S/'cells.json').read_text()).items() if k.startswith('FUSED')},**cells}
for spec in cells.values():
 cell=(R/spec['cell']).read_text();assert sha(cell)==spec['cell_sha256'] and cell in expanded
 cuda=next(ast.literal_eval(n.value) for n in ast.parse(cell).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CUDA'));assert sha(cuda)==spec['cuda_sha256']
for arm in json.loads((D/'manifest.json').read_text()):assert (D/(arm.lower()+'_selector.py.inc')).read_text() in expanded
if D.name in ('dequant','occupancy','overlap'):assert (S/'base_selector.py.inc').read_text() in expanded
assert result['job']==queue['job'] and result['measured'] and not result['invalid'];rec=result['seed_level_record'];assert not rec['failures'] and rec['seeds']==16
assert rec['weights_seed']==payload['weights_seed'] and rec['route_seed']==payload['route_seed'];assert len(rec['rows'])==payload['max_pairs'] and len(rec['AA_rows'])==48
for row in result['summary']:
 assert not row['invalid'] and row['seeds']==16
 row['CI95']=[row['ratio']-2.131449545559323*row['SE'],row['ratio']+2.131449545559323*row['SE']]
result['source_audit']=dict(verified=True,job=result['job'],wrapper_sha256=queue['sha256'],payload_sha256=payload['sha256'],exact_cells=len(cells),paired_rows=len(rec['rows']),AA_rows=len(rec['AA_rows']))
log=json.loads(receipt.read_text())['result']['log_tail'];packets=read_packets(log)
if packets:
 packet=next(v for v in packets if v['source_sha256']==payload['sha256'][:16]);assert packet['packet'].get('complete') is not False and not packet['packet'].get('transport_errors')
 result['source_audit']['packet_checksum']=packet['checksum']
line=next(l for l in result_lines(log) if l.startswith('RESOURCE '));result['resources']=ast.literal_eval(line[line.index(' [[8,')+1:])
(D/'results.json').write_text(json.dumps(result,indent=2)+'\n')
for v in result['summary']:print(v['arm'],v['control'],v['M'],round(v['ratio'],6),[round(x,6) for x in v['CI95']],v['point_nominee'])
