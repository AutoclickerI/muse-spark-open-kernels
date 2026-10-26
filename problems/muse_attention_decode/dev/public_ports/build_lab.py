"""Self-contained lab from reviewed source and exact S19, never donor drivers."""
import base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
import local_eval,h200_lab
b=json.loads((H.parent/'full_attention/phases/base_bundle.json').read_text())
b['operator']=(H/'out/operator.py').read_text();b['lee']=(H/'out/lee.py').read_text();b['input']=(H/'input.py.inc').read_text();b['notice']=(H/'NOTICES.txt').read_text()
p=R/'problems/muse_attention_decode';ref='\n'.join(l for l in (p/'reference.py').read_text().splitlines() if not l.startswith('from checks import'))
b['checker']='\n'.join(l for s in ((p/'checks.py').read_text(),ref) for l in s.splitlines() if not l.startswith('from __future__ import'))+'\nCACHE_CHUNK=8192\n'
b['official']=local_eval.tests_of((p/'TASK.md').read_text())
b['guard_cu']='#include <cuda_runtime.h>\n#include <cstdint>\n#include <vector>\n#include <stdexcept>\nint64_t graph_kernels(int64_t p){size_t n=0;auto g=(cudaGraph_t)p;auto e=cudaGraphGetNodes(g,nullptr,&n);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));std::vector<cudaGraphNode_t>x(n);cudaGraphGetNodes(g,x.data(),&n);int k=0;for(auto a:x){cudaGraphNodeType t;cudaGraphNodeGetType(a,&t);k+=t==cudaGraphNodeTypeKernel;}return k;}\n'
blob=base64.b85encode(lzma.compress(json.dumps(b,separators=(',',':')).encode(),preset=9)).decode()
s='import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types,sys\nimport torch\nBUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(blob)+')))\n'+(H/'adapter.py.inc').read_text()+(H/'lab.py.inc').read_text()
o=H/'out';(o/'public_attention.py').write_text(s)
meta=dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),seeds=[8671259+7919*i for i in range(8)],rounds=8,official_TASK=128,ABI_TASK=192,scored_TASK=384,identities={k:hashlib.sha256(b[k].encode()).hexdigest() for k in ('operator','lee','input')})
(o/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
wrapper,_=h200_lab.build([('public_attention.py',meta['payload_sha256'],s)],host='muse_bf16_proj',lab_only=True,import_budget=840)
(o/'preview.py').write_text(wrapper);print(meta)
