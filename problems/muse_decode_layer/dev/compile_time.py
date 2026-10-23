import sys, time, importlib.util, os, tempfile
os.environ["TORCH_EXTENSIONS_DIR"] = tempfile.mkdtemp(dir="/tmp")
import torch
repo = "/home/jihoon/competitions/muse_kernel/problems/"
for rel, fn in (a.split(":") for a in sys.argv[1:]):
    spec = importlib.util.spec_from_file_location("m", repo + rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    t0 = time.time()
    getattr(m, fn)()
    print(f"{rel}: cold compile {time.time() - t0:.1f} s")
