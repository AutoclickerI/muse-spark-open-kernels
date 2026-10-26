"""S17 local TASK/graph screen, with explicit stand-ins for Hopper launch geometry."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;S16=H.parent/'s16'
sys.path.insert(0,str(S16))
import screen as old
original=old.install


def install(m,forced=False):
    records=original(m,forced)
    if hasattr(m,'r2_attention'):
        attention=m.r2_attention
        def local_attention(data,qkv=None,kind='last',qt=1,ns=8):
            return attention(data,qkv,kind,qt,1 if kind in ('dsm','dsm_old') else ns)
        m.r2_attention=local_attention
        print('S17_LOCAL: exact P5 native math and allocation, genuine C1 launch instead of multi-SM DSM',flush=True)
    return records


old.install=install
if __name__=='__main__':old.main()
