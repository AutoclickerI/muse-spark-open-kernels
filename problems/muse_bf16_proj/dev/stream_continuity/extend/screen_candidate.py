"""Regular local_eval with S18's explicit Hopper stand-ins and current root preflight."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4];sys.path.insert(0,str(R/'tools'))
# S16's legacy screen directory also contains a screen_all.py. Resolve the
# current root tool before that directory is added by its inherited wrapper.
import screen_all
sp=importlib.util.spec_from_file_location('ew_prod_screen',R/'problems/muse_decode_layer/dev/s18/screen.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
m.old.main()
