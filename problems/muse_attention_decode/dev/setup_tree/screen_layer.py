"""Normal/forced S19 whole-layer screen with local DSM geometry only."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
import local_eval,kernels_new as k
sp=importlib.util.spec_from_file_location('p5_setup_layer_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');scr=importlib.util.module_from_spec(sp);sp.loader.exec_module(scr)
original=scr.install
def install(m,forced=False):
    records=original(m,forced);m.st_LOCAL=True;m.st_SOURCES={n:k.source(n[0],n[1],True) for n in m.st_SOURCES};return records
scr.old.install=install
if __name__=='__main__':scr.old.main()
