"""Pin S15n/e/L lane changes as independent asserted S15 patches."""
import hashlib,importlib.util,json
from pathlib import Path
import capture,patches
H=Path(__file__).resolve().parent


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    base=(H.parents[1]/'cand_r3_best7_s15.py').read_text();sources={}
    assert hashlib.sha256(base.encode()).hexdigest()=='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
    nearby=H.parents[1]/'cand_r3_best7_s15n.py';sources['nearby']=nearby.read_text()
    assert hashlib.sha256(sources['nearby'].encode()).hexdigest()=='ebceda91ed055adc0ece9161a350934dc92b93cd78277bc6b25e54ce66d890a6'
    release=module(H.parent/'pdl_release/build.py','s16_release_builder')
    for group in ('attention','moe','projections'):
        sources['release_'+group]=release.compose(tuple(g for g in ('attention','moe','projections') if g!=group))[0]
    library=module(H.parent/'library_chain/build.py','s16_library_builder')
    for group in library.GROUPS:sources['library_'+group]=library.compose(tuple(g for g in library.GROUPS if g!=group))
    cells={name:capture.delta(base,source) for name,source in sources.items()}
    target=H/'layer_cells.json';target.write_text(json.dumps(cells,separators=(',',':'))+'\n')
    meta=dict(base_sha256=patches.BASE_SHA,cells_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              source_sha256={name:hashlib.sha256(s.encode()).hexdigest() for name,s in sources.items()})
    (H/'layer_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print('Pinned',len(cells),'layer pieces')


if __name__=='__main__':main()
