"""Independent exact S19 patch containing only measured point winners."""
import argparse,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'out'
a=argparse.ArgumentParser();a.add_argument('--selection',type=Path,required=True);a.add_argument('--stem',default='candidate');a=a.parse_args();assert a.stem not in ('operator','lee','input'),'Reserved reviewed source stem'
sel=json.loads(a.selection.read_text());points=json.loads((H/'result.json').read_text());assert points['measured']
admit={(x['cell'],x['arm']) for x in points['points'] if x['r']<1};assert all((k,v) in admit for k,v in sel.items())
b=json.loads((H.parent/'full_attention/phases/base_bundle.json').read_text());source=b['source'];dispatch={tuple(map(int,k[:-1].split('x')))+(k[-1]=='s',):'operator' if v=='O' else 'lee' for k,v in sel.items()}
ports={n:(o/(n+'.py')).read_text() for n in sorted(set(dispatch.values()))};ports['input']=(H/'input.py.inc').read_text()
s=source+'\n_P5B_PUBLIC_NOTICES = '+repr((H/'NOTICES.txt').read_text())+'\n_PUBLIC_SELECTION='+repr(dispatch)+'\n_PUBLIC_SOURCES='+repr(ports)+'\n'+(H/'runtime.py.inc').read_text()
compile(s,'licensed_S19','exec');(o/(a.stem+'.py')).write_text(s)
patch=''.join(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='a/problems/muse_decode_layer/cand_r3_best7_s19_final.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s19_final.py'))
(H/(a.stem+'.patch')).write_text(patch);meta=dict(base_sha256=hashlib.sha256(source.encode()).hexdigest(),source_sha256=hashlib.sha256(s.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),selection=sel,notices_sha256=hashlib.sha256((H/'NOTICES.txt').read_bytes()).hexdigest(),scope='Only selected cells, S19 ABI, all other dispatch unchanged. P4 confirms actual full prepare and held-out G0.')
(H/(a.stem+'_manifest.json')).write_text(json.dumps(meta,indent=2)+'\n');print(meta)
