"""S18 local screen: S17's stand-ins, plus the same local VQ8 stand-in for no_Vwait32's off-row module.

sm120 cannot launch the Hopper VQ8 ring (its shared memory exceeds the 99 KiB limit); S15's stand-in
replaces _vq8_module with S14's N64 launch. The no_Vwait32 arm loads S17's VQ8 for verify 32 through its
own _s18_vwait_off_module, which needs the same local stand-in. Its exact CUDA is compiled for sm_90a by
audit_native.py and runs natively on the H200.
"""
import importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('s17_screen',H.parent/'s17'/'screen.py')
s17=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(s17)
old=s17.old;base=old.install


def install(m,forced=False):
    records=base(m,forced)
    if hasattr(m,'_s18_vwait_off_module') and m.torch.cuda.get_device_capability()!=(9,0):
        m._s18_vwait_off_module=m._vq8_module
        print('S18_LOCAL: no_Vwait32 off-row VQ8 uses the local VQ8 stand-in',flush=True)
    return records


old.install=install
if __name__=='__main__':old.main()
