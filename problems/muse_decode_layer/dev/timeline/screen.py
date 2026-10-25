"""Run the timeline payload plus local_eval's official decode tests on sm_120.

--forced exercises Hopper-only Python attention glue with the established
portable launch stand-ins. The GPU graph-query and CUPTI code is identical.
"""
import argparse
import json
import runpy
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev')]
import local_eval

parser=argparse.ArgumentParser()
parser.add_argument('--forced',action='store_true')
args=parser.parse_args()
mode='forced' if args.forced else 'normal'
old=local_eval.load_module
records={}

def loader(name,path):
    module=old(name,path)
    if name!='submission':return module
    module._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    if args.forced:
        import verify_best
        records['standins']=verify_best.install(module,False)
    prepared=module.prepare
    def prepare(weights):
        state=prepared(weights)
        ns=runpy.run_path(str(HERE/'lab_timeline.py'))
        buf=module.torch.empty(1<<30,dtype=module.torch.uint8,device='cuda')
        ns['lab'].__globals__.update(TL_EXISTING_HOST=(module,state),lab_cold_l2=lambda:buf.view(module.torch.int64).sum())
        try:
            lines=ns['lab'](time.monotonic()+260,weights)
        finally:
            results=ns['lab'].__globals__['TL_RESULTS']
            (HERE/(mode+'_timeline_local.json')).write_text(json.dumps(results,indent=2)+'\n')
        (HERE/(mode+'_timeline_local.json')).write_text(json.dumps(results,indent=2)+'\n')
        assert len(results)==12
        assert all(r['nodes'] for r in results)
        assert all(abs(r['wall']-r['busy_union']-r['bubble'])<1e-5 for r in results)
        assert all(r['bytes']>0 for r in results)
        records['complete_shapes']=len(results)
        records['archive_chars']=sum(len(line)+1 for line in lines)
        assert records['archive_chars']<9000
        print('\n'.join(lines),flush=True)
        (HERE/(mode+'_screen_local.json')).write_text(json.dumps(records,indent=2)+'\n')
        return state
    module.prepare=prepare
    return module
local_eval.load_module=loader
sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py'),
          '--tests-only','--json',str(HERE/(mode+'_checks_local.json'))]
local_eval.main()
