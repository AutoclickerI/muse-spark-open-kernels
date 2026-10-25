"""The same TASK and projection-partials checks for the 32-thread decode variant."""
import runpy
from pathlib import Path
import driver,build_warp
HERE=Path(__file__).resolve().parent
src=(HERE/'check.py').read_text()
src=src.replace("if '--quick' in sys.argv:CASES=CASES[:1]+CASES[2:3]+CASES[-6:-5]+CASES[-3:-2]", "CASES=[c for c in CASES if c.get('q_len',1)==1]")
driver.CUDA=build_warp.source();driver._MOD.clear()
exec(compile(src,str(HERE/'check.py'),'exec'),{'__name__':'__main__','__file__':str(HERE/'check.py')})
