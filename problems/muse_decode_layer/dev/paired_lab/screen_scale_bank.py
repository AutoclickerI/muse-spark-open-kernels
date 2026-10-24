"""Screen the scale-bank payload inside its actual board-host wrapper."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev')]
import local_eval
import verify_best


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--share-s',type=float,default=80.)
    args=ap.parse_args();load=local_eval.load_module;records={}
    def loader(name,path):
        module=load(name,path)
        if name=='submission':
            def run_one(label,source,deadline,weights,state):
                import linecache
                filename='<h200_lab '+label+'>'
                linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
                calls={}
                def install(m):calls.update(verify_best.install(m,False))
                ns['DP_LOCAL_INSTALL']=install
                started=time.monotonic()
                lines=ns['lab'](min(deadline,started+args.share_s),weights,state)
                rows=ns['DP_ROWS']
                records.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    rows=rows,lines=lines,elapsed_s=time.monotonic()-started,
                    forced_calls=calls,share_s=args.share_s)
                assert len(rows)==24,lines[-1]
                assert {r['case'] for r in rows}=={2,3,4,5,6,7,8,9}
                assert all(len([r for r in rows if r['case']==c])==3 for c in {r['case'] for r in rows})
                assert not any('REJECT' in line or 'INCOMPLETE' in line for line in lines),lines
                assert all(r['rounds']==6 and len(r['samples'][0])==6 for r in rows)
                assert sum(len(line)+1 for line in lines)<=4500
                assert ns['DP_MODULES']['best6'].V9_CUDA_SRC!=ns['DP_HOST_CUDA']
                (HERE/'scale_bank_rows_local.json').write_text(json.dumps(records,indent=2)+'\n')
                return lines
            module._lab_run_one=run_one
        return module
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/lab/lab_decode_scale_bank.py'),
              '--no-baseline','--json',str(HERE/'scale_bank_screen_local.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:assert len(records.get('rows',[]))==24,'payload failed despite host passing'
        raise


if __name__=='__main__':main()
