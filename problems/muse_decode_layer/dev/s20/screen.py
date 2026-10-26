"""S20's exact layer on GPU0, using S19's explicit Hopper stand-ins."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s19=load(H.parent/'s19/screen.py','s20_prior_screen')
portable=load(R/'problems/muse_bf16_proj/dev/dense_all/portable.py','s20_portable')
def install(m,forced=False):
    records=s19.install(m,forced)
    if hasattr(m,'DA_SELECTED'):
        m.DA_LOCAL_REDUCE=True;m.DA_WIDE_STANDIN=True;m.DA_LOCAL_MM=portable.mm
    if hasattr(m,'_s20_m64_attention') and not forced:
        # Install FA3's portable arithmetic; restore every candidate binding so
        # the normal screen keeps its normal branches. Only the external FA3
        # call inside the new M64 Python glue uses the explicit stand-in.
        saved=dict(vars(m));s19.verify_best.install(m,None)
        for k,v in saved.items():setattr(m,k,v)
        for k in set(vars(m))-set(saved):delattr(m,k)
    return records
if __name__=='__main__':
    s19.old.install=install;s19.old.main()
