from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;route=HERE.parent/'route_glue'
source='RG_CUDA='+repr((route/'route_once.cu').read_text())+'\n'+(route/'runtime.py.inc').read_text()
source+='\n'+(HERE/'dropin.py').read_text()+'\n'+(HERE/'adapter.py.inc').read_text()
(HERE/'best7_coop_glue_cell.py').write_text(source)
print('cell',hashlib.sha256(source.encode()).hexdigest())
