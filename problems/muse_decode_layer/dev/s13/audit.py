"""S13 complete retained-state audit and local_eval normal/forced screen."""
from pathlib import Path
import runpy
import sys
H=Path(__file__).resolve().parent
if '--out' not in sys.argv:sys.argv.extend(['--out',str(H)])
runpy.run_path(str(H.parent/'s12/audit.py'),run_name='__main__')
