"""Build and serially screen all reset implementations; failures stay local."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import patches

H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('names',nargs='*');a=ap.parse_args()
    names=a.names or list(patches.registry())
    admission=json.loads((H/'admission.json').read_text()) if (H/'admission.json').exists() else {}
    for name in names:
        source,meta=patches.compose([name]);candidate=H/'generated/variants'/f'{name}.py';candidate.parent.mkdir(exist_ok=True,parents=True);candidate.write_text(source)
        entry=admission.setdefault(name,{})
        for mode in ('normal','forced'):
            if entry.get(mode,{}).get('sha256')==meta['sha256'] and entry[mode].get('passed'):continue
            out=H/'screen'/name/mode;out.mkdir(parents=True,exist_ok=True)
            cmd=[str(R/'tools/img'),'python',str(H/'screen.py'),str(candidate),'--out',str(out)]+(['--forced'] if mode=='forced' else [])
            print('START',name,mode,meta['sha256'][:8],flush=True)
            with (out/'screen.log').open('w') as log:
                result=subprocess.run(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT)
            checks=json.loads((out/'local.json').read_text()) if (out/'local.json').exists() else {}
            ok=result.returncode==0 and checks.get('sha256')==meta['sha256'] and all(
                checks.get(key) and all(row['ok'] for row in checks[key]) for key in ('tests','shapes','static'))
            entry[mode]=dict(sha256=meta['sha256'],passed=ok,exit_code=result.returncode,
                counts={k:sum(bool(x['ok']) for x in checks.get(k,[])) for k in ('tests','shapes','static')},
                local_report=str((out/'local.json').relative_to(H)))
            (H/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
            print('PASS' if ok else 'FAIL',name,mode,entry[mode]['counts'],flush=True)
            if not ok:raise RuntimeError(str(out/'screen.log'))


if __name__=='__main__':main()
