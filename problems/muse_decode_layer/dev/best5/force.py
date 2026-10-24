"""Force best's Hopper glue and NT2 full-layer graphs on the local sm120 GPU."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'problems/muse_decode_layer/dev'))
import local_eval
import verify_best


def install(mod,cold):
    verify_best.install(mod,cold)
    original=mod.prepare
    def prepare(weights):
        state=original(weights)
        assert all(row['reason']=='measured' for row in state['moe_nt_rows'].values()),state['moe_nt_rows']
        # The regular tuner gates both real native kernels over three changed
        # input sets. Force NT2 on every eligible full-layer scored graph too.
        state['moe_nt_plan']={m:2 for m in (32,128,256)}
        print('BEST5_FORCED_NT',json.dumps(state['moe_nt_plan']),flush=True)
        return state
    mod.prepare=prepare


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cold-native',action='store_true');args=ap.parse_args()
    original=local_eval.load_module
    def load(name,path):
        mod=original(name,path)
        if name=='submission':install(mod,args.cold_native)
        return mod
    local_eval.load_module=load
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/cand_r3_best5.py'),'--no-baseline',
              '--json',str(ROOT/'problems/muse_decode_layer/dev/best5_forced_local.json')]
    local_eval.main()

if __name__=='__main__':main()
