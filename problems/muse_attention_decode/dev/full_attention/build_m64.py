"""One new-piece handoff: no other S19 dispatch or compiled kernel changes."""
import difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'phases';b=json.loads((o/'base_bundle.json').read_text())
s=b['source']+'\n'+(H/'m64_driver.py.inc').read_text();compile(s,'S19_M64','exec');(o/'m64.py').write_text(s)
patch=''.join(difflib.unified_diff(b['source'].splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s19_final.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s19_final.py'))
(o/'m64.patch').write_text(patch);meta=dict(base_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),source_sha256=hashlib.sha256(s.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),addition_lines=len((H/'m64_driver.py.inc').read_text().splitlines()),selection={'1x8f':'M64'},cuda_units_changed=0,prepare='17 int32 literal values; no input tensors or runtime lengths',receipt='deea291e294b4a53')
(o/'m64_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
