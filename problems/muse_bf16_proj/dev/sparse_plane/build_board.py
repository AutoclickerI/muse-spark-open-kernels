from pathlib import Path
import hashlib
H=Path(__file__).resolve().parent
notice=(H.parent/'public_nf/LICENSE.NathenFernandes').read_text()
s='# okc: problem=muse_bf16_proj\n'+'\n'.join('# '+l for l in ('Nathen Fernandes; github.com/NathenFernandes/muse-bf16-proj @ 86aff75; MIT\n'+notice).splitlines())+'\n'+(H/'adapter.py').read_text()+'\n'+(H/'board.py.inc').read_text()
compile(s,'<sparse board screen>','exec');(H/'candidate.py').write_text(s);print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
