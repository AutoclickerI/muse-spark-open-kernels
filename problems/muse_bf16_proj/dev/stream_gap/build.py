"""Build direct coded-output experiments and a minimal MIT router port."""
import ast,base64,zlib,hashlib,json,builtins
from pathlib import Path
from component_source import projection_source
H=Path(__file__).resolve().parent;R=H.parents[3]
# The selected, attributed definitions are retained here. Rebuilding never needs
# the previous agent's temporary expanded upstream checkout.
provenance=json.loads((H/'manifest.json').read_text())
router=(H/'router_port.py').read_text()
assert hashlib.sha256(router.encode()).hexdigest()==provenance['router_sha256']
compile(router,'<router_port>','exec')
base=projection_source()
assert hashlib.sha256(base.encode()).hexdigest().startswith('2d5caebc')
body='\nimport base64,zlib\nSG_ROUTER_SOURCE=zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(router.encode(),9)).decode())+')).decode()\n'+(H/'extension.py.inc').read_text();out=base+body
compile(out,'<stream-gap>','exec');assert len(out.encode())<1000000,len(out)
(H/'candidate.py').write_text(out)
assert hashlib.sha256(out.encode()).hexdigest()==provenance['source_sha256']
print('candidate',len(out.encode()),hashlib.sha256(out.encode()).hexdigest()[:8], 'router',len(router))
