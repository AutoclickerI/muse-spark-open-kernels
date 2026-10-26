"""Local S19 screens: S18's stand-ins, plus P5's explicit local two-WG sources for the wr (RAW2/PIPE2) kernels."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def _load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
# dev/s16 leads sys.path in S16's screen and has its own screen_all; tools/local_preflight needs tools' one.
sys.modules['screen_all']=_load(R/'tools/screen_all.py','screen_all')
s18=_load(H.parent/'s18/screen.py','s18_screen')
WIN=R/'problems/muse_attention_decode/dev/window_rebuild';sys.path.insert(0,str(WIN))
wr_local=_load(WIN/'kernel.py','s19_wr_local_kernel')
old=s18.old;base=old.install
s16_load=old.load
sys.path.insert(0,str(H.parent));import verify_best


def forced_hopper(m,cold,exercise_table=False,exercise_prefetch=False):
    """S19 is baked: no attention tuner, candidate list or prefill adapters. Force only the Hopper branches (FA3,
    the port kernels, Lt) with verify_best's stand-ins and screen the plans the candidate ships."""
    prepare=m.prepare;calls=verify_best.install(m,cold);m.prepare=prepare
    print('S19_FORCED_HOPPER FA3/port/Lt stand-ins; shipped plans',flush=True)
    return calls


def load(path,name):
    if Path(path).as_posix().endswith('best7/force.py'):
        import types;return types.SimpleNamespace(install=forced_hopper)
    return s16_load(path,name)


old.load=load


def install(m,forced=False):
    records=base(m,forced)
    if hasattr(m,'wr_KERNELS') and m.torch.cuda.get_device_capability()!=(9,0):
        m.wr_KERNELS={n:wr_local.source(c,local=True,stream=n.startswith('STREAM'),pipeline='WINDOW_STEP' in m.wr_KERNELS[n]) for n,c in m.wr_TOPOLOGY.items() if n in m.wr_KERNELS}
        print('S19_LOCAL: wr two-WG kernels use P5 local sources (global last-WG publication replaces DSM)',flush=True)
    return records


old.install=install
if __name__=='__main__':old.main()
