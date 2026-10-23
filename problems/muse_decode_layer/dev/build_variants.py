from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from vendor import vendor
root=Path(__file__).resolve().parents[1]
s=(root/'cand_glue.py').read_text()
a=s.index('# Losslessly vendored MoE v4');b=s.index('\n\n# LIDS already encodes',a)
s=s[:a]+'# MoE v7: incremental expert/tile iterator, same lossless layout.\n'+vendor(root.parent/'muse_moe_w4a16/cand_v7.py','mv4_')+s[b:]
(root/'cand_moe7.py').write_text(s)
s=(root/'cand_native.py').read_text()
a=s.index('def _chunks(tokens):');b=s.index('\n\ndef boundary(',a)
s=s[:a]+'''def _chunks(tokens):
    return 16 if tokens <= 8 else 1
'''+s[b:]
s=s.replace('num_warps=16 if C == 1 else 4','num_warps=8 if C == 1 else 4')
s=s.replace('sync = torch.zeros((15 * tokens,), device=dev, dtype=torch.int32)', 'sync = (torch.empty if _chunks(tokens) == 1 else torch.zeros)((15 * tokens,), device=dev, dtype=torch.int32)')
s=s.replace('if cfg["q_len"] == 1 and cfg["batch"] > 1:', 'if cfg["q_len"] == 1:')
(root/'cand_native_glue.py').write_text(s)
