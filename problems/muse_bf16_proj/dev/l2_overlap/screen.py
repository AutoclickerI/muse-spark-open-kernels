"""Execute both transports and the complete lab on the local GPU, without skips."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev')]
import local_eval
import verify_best


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--share-s',type=float,default=80.)
    args=parser.parse_args();original=local_eval.load_module;records={}
    (HERE/'validation.json').unlink(missing_ok=True)
    def load(name,path):
        module=original(name,path)
        if name=='submission':
            def run(label,source,deadline,weights,state):
                import linecache
                filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
                ns['P7_LOCAL_INSTALL']=lambda m:verify_best.install(m,False)
                started=time.monotonic();lines=ns['lab'](min(deadline,started+args.share_s),weights,state)
                rows=ns['P7_LOCAL_ROWS'];records.update(rows=rows,lines=lines,elapsed_s=time.monotonic()-started)
                (HERE/'rows_local.json').write_text(json.dumps(records,indent=2)+'\n')
                assert len({r['case'] for r in rows})==6,lines
                assert len(rows)==12,lines
                assert not any(any(word in line for word in ('INCOMPLETE','REJECT','NO_PICK','ERROR')) for line in lines),lines
                assert len([l for l in lines if l.startswith('P7_ST ')])==6
                assert sum(len(l)+1 for l in lines)<=4500
                records['admitted']=True
                return lines
            module._lab_run_one=run
        return module
    local_eval.load_module=load
    host=ROOT/'problems/muse_prefill_layer/lab/lab_l2_overlap_lab.py'
    sys.argv=[sys.argv[0],str(host),'--tests-only','--no-baseline',
              '--json',str(HERE/'host_local.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:
            assert len(records.get('rows',[]))==12,'payload failed inside the passing host'
            assert records.get('admitted') is True,'payload guard failed inside the passing host'
            payload=HERE.parent/'l2_overlap_lab.py'
            report=dict(payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                        host_sha256=hashlib.sha256(host.read_bytes()).hexdigest(),
                        share_s=args.share_s,elapsed_s=records['elapsed_s'],cases=6,final_pairs=12,
                        profiles=6,host_tests=5,log_characters=sum(len(l)+1 for l in records['lines']))
            (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        raise


if __name__=='__main__':main()
