"""Native cluster gang-size sweep after last-CTA verify proved serialized."""
import hashlib,re
from pathlib import Path
import build_attention_board,build_last,build
HERE=Path(__file__).resolve().parent
s=(HERE/'out/p5_global_board.py').read_text()
for prefix,path,cuda in [('lc_',HERE/'last_driver.py',build_last.source()),('cc_',HERE/'driver.py',build.source())]:
    cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
    s+='\n'+build_attention_board.driver_cell(path,prefix,cuda,cpp)
s+='\n'+(HERE/'dsm_probe.py.inc').read_text();compile(s,'p5_dsm_board.py','exec');p=HERE/'out/p5_dsm_board.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s))
