import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];O=H/'out';sys.path.insert(0,str(R/'tools'))
import h200_lab
b=json.loads((O/'bundle.json').read_text())
s19=(R/'problems/muse_decode_layer/dev/s19/generated/s19.py').read_text();assert hashlib.sha256(s19.encode()).hexdigest().startswith('62ea5bf4')
b['s19']=s19
problem=R/'problems/muse_decode_layer'
ref='\n'.join(x for x in (problem/'reference.py').read_text().splitlines() if not x.startswith(('from checks import','from nvfp4 import')))
checker='\n'.join(x for t in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),ref) for x in t.splitlines() if not x.startswith('from __future__ import'))
ports=((16,1,True),(32,1,True),(1,8,True),(1,8,False),(16,8,True),(16,8,False),(32,8,True))
cases=tuple((b,l,w) for l in (1,8) for b in (1,16,32) for w in (2048,0))
seeds=tuple(281730011+104729*i for i in range(8))
identities=dict(control=hashlib.sha256(b['base'].encode()).hexdigest(),S19=hashlib.sha256(s19.encode()).hexdigest(),**{n:hashlib.sha256(s.encode()).hexdigest() for n,s in b['modules'].items()},native_cs=hashlib.sha256(b['native_cs'].encode()).hexdigest(),native_partial=hashlib.sha256(b['native_partial'].encode()).hexdigest(),TASK=hashlib.sha256(checker.encode()).hexdigest())
encoded=base64.b85encode(lzma.compress(json.dumps(b,separators=(',',':')).encode(),preset=9)).decode()
source='LAB_RESOURCES={"compile_s":400,"run_s":440,"output_bytes":11700,"context_poison":"none"}\nimport ast,base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\n'
source+='BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(encoded)+')))\nCHECKER='+repr(checker)+'\nPORTS='+repr(ports)+'\nCASES='+repr(cases)+'\nSEEDS='+repr(seeds)+'\nIDENTITIES='+repr(identities)+'\nWEIGHT_SEED=281730007\nROUTER_SEED=281730009\nROUNDS=8\nCTX=80000\n'
helpers=(R/'problems/muse_decode_layer/dev/s19/graph_helpers.py.inc').read_text()
tree=ast.parse(helpers)
source+='\n'.join(ast.get_source_segment(helpers,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('ab_copy','ab_graph'))+'\n'
source+='''
if 'lab_cold_l2' not in globals():
 _PV_FLUSH=None
 def lab_cold_l2():
  global _PV_FLUSH
  if _PV_FLUSH is None:_PV_FLUSH=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
  _PV_FLUSH.view(torch.int64).sum()
'''
source+=(H.parent/'public_hybrid/codec.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text()
compile(source,'in_graph','exec');(O/'in_graph.py').write_text(source)
wrapper,_=h200_lab.build([('in_graph',hashlib.sha256(source.encode()).hexdigest()[:16],source)],host='muse_attention_decode',lab_only=True,import_budget=840.0)
(O/'wrapper.py').write_text(wrapper)
meta=dict(identities=identities,payload_sha256=hashlib.sha256(source.encode()).hexdigest(),wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),seeds=seeds,weight_seed=281730007,router_seed=281730009,rounds=8,cases=cases,ports=ports,expected_TASK=8*(7*6+2*3+3*2),host='muse_attention_decode',import_budget=840,decision='in-graph attention transfer and vectorized direct partials')
(H/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['payload_sha256'],len(wrapper),meta['expected_TASK'])
