"""Reference/cache/fresh graph checks for the cooperative decode prep replacement."""
from pathlib import Path
import driver,build_compact
HERE=Path(__file__).resolve().parent
driver.CUDA=build_compact.source();driver._MOD.clear()
src=(HERE/'check.py').read_text().replace("if '--quick' in sys.argv:CASES=CASES[:1]+CASES[2:3]+CASES[-6:-5]+CASES[-3:-2]", "CASES=[c for c in CASES if c.get('q_len',1)==1]")
exec(compile(src,str(HERE/'check.py'),'exec'),{'__name__':'__main__','__file__':str(HERE/'check.py')})
