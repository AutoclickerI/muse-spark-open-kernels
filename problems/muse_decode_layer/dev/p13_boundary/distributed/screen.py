"""Full local TASK and graph screen; only unavailable Hopper execution substituted."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path.insert(0,str(R/'tools'))
import local_eval

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);ap.add_argument('--name',default='boundary_A');a=ap.parse_args()
    kind='route' if a.candidate.name.startswith('route_') else 'boundary'
    series=json.loads((H/(kind+'_manifest.json')).read_text())['base_series'].lower()
    spec=importlib.util.spec_from_file_location('p13_production_screen',H.parents[1]/series/'screen.py')
    # H.parents[1] is dev/, whereas H.parent is the P13 evidence directory.
    scr=importlib.util.module_from_spec(spec);spec.loader.exec_module(scr)
    original=local_eval.load_module;records={}
    def loader(name,path):
        m=original(name,path)
        if name=='submission':
            records['calls']=scr.install(m,True)
            if hasattr(m,'P13B_LOCAL'):m.P13B_LOCAL=True
        return m
    local_eval.load_module=loader
    out=H/'out'/a.name;out.mkdir(exist_ok=True)
    sys.argv=['local_eval.py',str(a.candidate),'--no-baseline','--reps','4','--inputs','3','--json',str(out/'local.json')]
    try:local_eval.main()
    finally:(out/'calls.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
