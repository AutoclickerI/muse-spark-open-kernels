"""One S18+R8+D attention batch, with a labelled free-bound oracle."""
import base64,hashlib,json,lzma,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'fixed'
b=json.loads((o/'base_bundle.json').read_text());b['adapter']=(H/'attention_adapter.py.inc').read_text()
for name in ('A_vq','F_vq','B_vq','last','C3','C4'):b[name]=(o/(name+'.cu')).read_text()
# C++ prototypes derive from the exact native ABI, including fixed's extra ptr.
for name in ('F_vq','last'):
    funcs=('vq8_forward','vq8_resources') if name=='F_vq' else ('sliding_forward','sliding_resources')
    b[name+'_CPP']='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',b[name],re.S).group(0)+';' for fn in funcs)
p=R/'problems/muse_attention_decode'
ref='\n'.join(l for l in (p/'reference.py').read_text().splitlines() if not l.startswith('from checks import'))
b['checker']='\n'.join(l for s in ((p/'checks.py').read_text(),ref) for l in s.splitlines() if not l.startswith('from __future__ import'))+'\nCACHE_CHUNK=8192\n'
blob=base64.b85encode(lzma.compress(json.dumps(b,separators=(',',':')).encode(),preset=9)).decode()
payload='import ast,base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\n_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(blob)+')))\n'+(H/'fixed_lab.py.inc').read_text()
(o/'fixed_attention.py').write_text(payload)
meta=dict(base=json.loads((o/'base_meta.json').read_text()),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),
          native={n:hashlib.sha256(b[n].encode()).hexdigest() for n in ('A_vq','F_vq','B_vq','last','C3','C4')},
          seeds=[3671223+7919*i for i in range(8)],rounds=8,official_TASK=84,scored_TASK=320,stress_TASK=160,oracle='Ffree excludes its exact native K-only prepass; never nominatable. Fpaid recomputes bound inside every call.')
(o/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
