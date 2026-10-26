"""Concrete P4 patch onto S18+R8+D; only TASK-passing paid arms may ship."""
import argparse,difflib,hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'fixed'
ap=argparse.ArgumentParser();ap.add_argument('--selection',required=True,help='JSON mapping BxL to measured recipe');a=ap.parse_args()
b=json.loads((o/'base_bundle.json').read_text());selection={tuple(map(int,k.split('x'))):v for k,v in json.loads(a.selection).items()}
assert selection and all(v in ('C3','C4') for v in selection.values())
def cpp(c,functions):return '#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',c,re.S).group(0)+';' for fn in functions)
base=b['source']+'\n# P5b measured R8+D control.\nATTN_PORT_CUDA_SRC='+repr(b['q1'])+'\nVQ8_CUDA='+repr(b['vq'])+'\n'
(o/'control.py').write_text(base);sources={}
for role in sorted(set(selection.values())):
    c=(o/(role+'.cu')).read_text()
    sources[role]=(c,b['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])
source=base+'\n_P5_FIXED_SOURCES='+repr(sources)+'\n_P5_FIXED_SELECTION='+repr(selection)+'\n'+(H/'fixed_driver.py.inc').read_text()
(o/'candidate.py').write_text(source);patch=''.join(difflib.unified_diff(base.splitlines(True),source.splitlines(True),fromfile='a/S18_R8_D.py',tofile='b/S18_R8_D.py'))
(o/'candidate.patch').write_text(patch)
meta=dict(base_file='problems/muse_decode_layer/cand_r3_best7_s18_final.py',base_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),control_sha256=hashlib.sha256(base.encode()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),selection={f'{k[0]}x{k[1]}':v for k,v in selection.items()},gate='stage nomination; P4 confirms held-out whole-layer G0')
(o/'candidate_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
