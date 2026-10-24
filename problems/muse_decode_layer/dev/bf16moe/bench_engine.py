"""Run the prepare-only selector on the local GPU; never an H200 speed claim."""
import json
from pathlib import Path
import time
import check_engine as check
import deep_gemm
b=check.base
p=Path(__file__).parent/'prepare_tail.py'
s=p.read_text().split('_bf_base_prepare=prepare')[0]
exec(compile(s,str(p),'exec'),b.__dict__)
b._bf_fixture_make_fixture=check.moe_tune.make_fixture
w=check.ref.generate_weights(5353)
native=b.v9_prepare({**w,'post_norm_w':w['post_expert_norm']})
bank=b._bf_make_bank(w);bank['dg']=deep_gemm
choices,times=b._bf_tune(w,native,bank,time.monotonic()+180.)
Path('runs/screen/bf16-engine-tune.json').write_text(json.dumps(dict(choices=choices,us=times),indent=2)+'\n')
