"""Fresh-input/reference checks and repeated graph replay exercise counter reset and partial publication."""
from pathlib import Path
import driver,masked_driver
HERE=Path(__file__).resolve().parent
original=masked_driver.attention

def call(*a,**kw):
    out=original(*a,**kw)
    return out

driver.attention=call
src=(HERE/'check.py').read_text().replace("if '--quick' in sys.argv:CASES=CASES[:1]+CASES[2:3]+CASES[-6:-5]+CASES[-3:-2]", "CASES=[c for c in CASES if c.get('q_len',1)==1]")
src=src.replace('copy_into(data,fresh);graph.replay();torch.cuda.synchronize()', 'copy_into(data,fresh)\n            for _ in range(8):graph.replay()\n            torch.cuda.synchronize()\n            assert all(int(torch.count_nonzero(c))==0 for c in masked_driver._COUNTERS.values())')
exec(compile(src,str(HERE/'check.py'),'exec'),{'__name__':'__main__','__file__':str(HERE/'check.py'),'masked_driver':masked_driver})
