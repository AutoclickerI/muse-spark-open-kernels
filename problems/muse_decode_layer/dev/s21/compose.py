"""S21: exact S19 plus C3, M64 and P10's exact12 gate; overnight batch additions live here."""
import importlib.util,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s20=load(H.parent/'s20/compose.py','s21_s20_compose')
s19=s20.s19;sha=s20.sha;BASE_SHA=s20.BASE_SHA
DEFAULT=('C3','M64','GATE')
ROWS={p:s20.ROWS[p] for p in DEFAULT}
base=s20.base;expand=s20.expand

def build(enabled=DEFAULT):
    enabled=tuple(enabled);assert set(enabled)<=set(DEFAULT)
    source,meta=s20.build(enabled)
    meta.update(enabled=[p for p in DEFAULT if p in enabled],series='S21')
    return source,meta

if __name__=='__main__':
    (H/'generated').mkdir(exist_ok=True)
    s,m=build();(H/'generated/s21.py').write_text(s);(H/'selection.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
