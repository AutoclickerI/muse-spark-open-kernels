"""Read-only harness with real-width expert fixtures and the actual reader."""
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'serving/harness'))
import common
import run

tp=int(sys.argv[sys.argv.index('--tp')+1]) if '--tp' in sys.argv else 1
for kind in ('margin','random'):
    common.WEIGHT_SETS[kind]=(f'muse-bank-width-tp{tp}-{kind}',f'dflash-bank-width-{kind}')
base_launch,base_ready=run.launch,run.wait_ready


def launch(args,tree,out,port):
    args.mem_fraction_static=.3;args.max_total_tokens=300000
    if args.tp==2:args.env+=['SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0']
    if tree!=common.STOCK_TREE:
        args.env += [f"PYTHONPATH={tree/'python'}:{ROOT/'serving/tests/shared_reader_hook'}:{ROOT/'serving/harness/pyhook'}"]
    return base_launch(args,tree,out,port)


def ready(proc,port,log,timeout):
    elapsed=base_ready(proc,port,log,timeout)
    assert set(re.findall(r'max_total_num_tokens=(\d+)',Path(log).read_text()))=={'300000'}
    return elapsed


run.launch,run.wait_ready=launch,ready
if __name__=='__main__':
    if '--forced' in sys.argv:
        sys.argv.remove('--forced')
        import teacher_forced_workload
        teacher_forced_workload.main()
    else:run.main()
