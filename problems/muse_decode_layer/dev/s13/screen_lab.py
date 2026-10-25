"""Full wrapper lab followed by local_eval admission; exact unchanged Hopper compiles are reused."""
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parent/'s12')]
import local_eval
import compile_cache
compile_cache.install(local_eval)
sys.argv=['local_eval.py',str(R/'problems/muse_decode_layer/lab/lab_decode_s13ab.py'),
          '--tests-only','--json',str(H/'lab_local.json')]
local_eval.main()
