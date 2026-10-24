"""Exercise every Hopper Python family and NT2; only attention launches use stand-ins."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev')]
import local_eval
import verify_best

def install(mod,cold):
    verify_best.install(mod,cold)
    original=mod.prepare
    def prepare(weights):
        state=original(weights)
        state['moe_nt_plan']={t:2 for t in (32,128,256)}
        # This candidate's complete timed path is regular projection plus glue.
        state['prefix_plan']={t:('regular',True) for t in mod._TUNE_TOKENS}
        assert all(row['reason']=='measured' for row in state['moe_nt_rows'].values())
        print('GLUE_FORCED_NT2',json.dumps(state['moe_nt_plan']),flush=True)
        return state
    mod.prepare=prepare

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cold-native',action='store_true');a=ap.parse_args()
    old=local_eval.load_module
    def load(name,path):
        m=old(name,path)
        if name=='submission':install(m,a.cold_native)
        return m
    local_eval.load_module=load
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/cand_r3_best5_glue.py'),'--no-baseline',
              '--json',str(ROOT/'problems/muse_decode_layer/dev/glue_forced_local.json')]
    local_eval.main()
if __name__=='__main__':main()
