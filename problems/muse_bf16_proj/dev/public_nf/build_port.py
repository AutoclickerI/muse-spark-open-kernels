import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'port.py').read_text().replace('NF_VENDOR = None','NF_VENDOR = '+repr((H/'nf_selected.py').read_text()))
(H/'port_body.py').write_text(s)
print('port body',len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
