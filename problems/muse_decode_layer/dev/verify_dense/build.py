from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
s='VD_LT_CUDA='+repr((HERE/'lt.cu').read_text())+'\nVD_INC_CUDA='+repr((HERE/'incumbent_lt.cu').read_text())+'\nVD_CUTLASS_HOST='+repr((HERE/'cutlass_host.cpp').read_text())+'\n'+(HERE/'host_binary.py.inc').read_text()+'\n'+(HERE/'cubin.py.inc').read_text()+'\n'+(HERE/'runtime.py.inc').read_text()+'\n'+(HERE/'cubin_runtime.py.inc').read_text()
(HERE/'projection.py').write_text(s)
if (HERE/'adapter.py.inc').exists():(HERE/'best7_verify_dense_cell.py').write_text(s+'\n'+(HERE/'adapter.py.inc').read_text())
print(hashlib.sha256(s.encode()).hexdigest())
