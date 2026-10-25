"""Full wrapper lab followed by local_eval admission; exact unchanged Hopper compiles are reused."""
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parent/'s12')]
import local_eval
import compile_cache
compile_cache.install(local_eval)
target=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else R/'problems/muse_decode_layer/lab/lab_decode_s14ab.py'
output=H/('lab_local.json' if target.name=='lab_decode_s14ab.py' else target.stem+'_local.json')
sys.argv=['local_eval.py',str(target),'--tests-only','--json',str(output)]
local_eval.main()
