"""A native attention-board probe, while original-S11 layer-host quotas are exhausted."""
import hashlib,importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'cluster_sliding'));import build_attention_board
q=importlib.util.spec_from_file_location('_p5_one_head_src',HERE/'build.py');m=importlib.util.module_from_spec(q);q.loader.exec_module(m)
s=(HERE.parent/'cluster_sliding/out/p5_global_board.py').read_text()+'\n'+build_attention_board.driver_cell(HERE/'driver.py','h1_',m.source())+'\n'+(HERE/'board_probe.py.inc').read_text()
compile(s,'p5_native_head_board.py','exec');p=HERE/'out/p5_native_head_board.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s))
