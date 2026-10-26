"""Build the six-arm exact coded QKV geometry experiment, with attribution."""
from pathlib import Path
from component_source import projection_source
import ast,base64,hashlib,json,zlib
H=Path(__file__).resolve().parent;R=H.parents[3]
base=projection_source()
s=base+'\n'+(H/'qkv_dispatch.py.inc').read_text()
compile(s,'<sgq candidate>','exec');(H/'qkv_candidate.py').write_text(s)
notice=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text().split('import base64,zlib')[0]
reference=(R/'problems/muse_bf16_proj/reference.py').read_text()
names={'FP32_LAYERS','REL_L2_TOL','ROUTER_TOL','PROBE_TOL','ref_kernel','probe_rows','check_implementation'}
nodes=[n for n in ast.parse(reference).body if getattr(n,'name',None) in names or isinstance(n,ast.Assign) and any(getattr(t,'id',None) in names for t in n.targets)]
checker=(R/'problems/muse_bf16_proj/checks.py').read_text()+'\n'+'\n\n'.join(ast.get_source_segment(reference,n) for n in nodes)
p=notice+'\nimport torch,base64,zlib\nSGQ_SOURCE=zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(s.encode(),9)).decode())+')).decode()\nSGQ_CHECK_SOURCE='+repr(checker)+'\n'+(H/'qkv_lab.py.inc').read_text()
compile(p,'<sgq lab>','exec');(H/'p10_qkv_dispatch.py').write_text(p)
(H/'qkv_manifest.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),payload_sha256=hashlib.sha256(p.encode()).hexdigest(),base_sha256=hashlib.sha256(base.encode()).hexdigest()),indent=2)+'\n')
# Keep the exact measured host code, compressed only for the server's 1MB source
# limit. linecache preserves Triton's source discovery after decompression.
host='# okc: problem=muse_bf16_proj\n'+notice+'\nimport base64,zlib,linecache\n_sgq_host_source=zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(base.encode(),9)).decode())+')).decode()\n_sgq_host_file="<P10 exact P12 host>"\nlinecache.cache[_sgq_host_file]=(len(_sgq_host_source),None,_sgq_host_source.splitlines(True),_sgq_host_file)\nexec(compile(_sgq_host_source,_sgq_host_file,"exec"),globals())\n_sgq_host_prepare=globals()["prepare"]\n_sgq_host_kernel=globals()["kernel_fn"]\ndef prepare(weights):\n    return _sgq_host_prepare(weights)\ndef kernel_fn(data,state):\n    return _sgq_host_kernel(data,state)\n'
(H/'compact_host.py').write_text(host)
print(len(p),hashlib.sha256(p.encode()).hexdigest()[:12])
