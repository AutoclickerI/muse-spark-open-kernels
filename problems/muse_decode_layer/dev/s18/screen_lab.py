"""One import transport smoke. Full held-out confirmation runs only on the H200."""
import argparse,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'tools'))
from local_eval import wrapper_smoke

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');args=ap.parse_args()
    result=wrapper_smoke('import');result['forced_flag']=args.forced
    print(json.dumps(result))
