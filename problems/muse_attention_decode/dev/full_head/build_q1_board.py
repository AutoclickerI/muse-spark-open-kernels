"""Native B1 full decode portfolio plus a paired kernel-only probe while layer quota waits."""
import hashlib,importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cluster_sliding'));import build_attention_board
q=importlib.util.spec_from_file_location('_p5_one_head_q1_src',HERE/'build_q1.py');m=importlib.util.module_from_spec(q);q.loader.exec_module(m)
s=(HERE.parent/'cluster_sliding/out/p5_global_board.py').read_text()+'\n'+build_attention_board.driver_cell(HERE/'q1_driver.py','h1_',m.source())+'\n'+(HERE/'q1_probe.py.inc').read_text()
compile(s,'p5_native_q1_board.py','exec');p=HERE/'out/p5_native_q1_board.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s));print('CUDA',hashlib.sha256(m.source().encode()).hexdigest())
