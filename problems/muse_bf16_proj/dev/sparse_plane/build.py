"""Build self-contained native sparse-plane and fused-pool experiments."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'public_nf'
s=(P/'port.py').read_text().replace('NF_VENDOR = None','NF_VENDOR = '+repr((P/'nf_selected.py').read_text()))
s+='\n'+(H/'sparse.py').read_text()+'\n'+(H/'adapter.py.inc').read_text()
compile(s,'<NF sparse plane adapter>','exec');(H/'adapter.py').write_text(s)
print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
