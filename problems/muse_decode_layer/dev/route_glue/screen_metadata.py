"""Exercise the exact lab wrapper with all Hopper Python paths forced locally."""
from pathlib import Path
import hashlib
import json
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(HERE.parent)]
import local_eval
import verify_best


def main():
    original=local_eval.load_module;record={}
    def loader(name,path):
        module=original(name,path)
        if name=='submission':
            def run_one(label,source,deadline,weights,state):
                import linecache
                filename='<h200_lab '+label+'>'
                linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
                current=(HERE/'dropin.py').read_text();current_file='<route_metadata_current>'
                linecache.cache[current_file]=(len(current),None,current.splitlines(True),current_file)
                exec(compile(current,current_file,'exec'),ns)
                calls={}
                def install(m):
                    nonlocal calls
                    calls=verify_best.install(m,False)
                ns['RG_LOCAL_INSTALL']=install
                started=time.monotonic();lines=ns['lab'](deadline,weights,state)
                record.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=ns['RG_ROWS'],
                    lines=lines,elapsed_s=time.monotonic()-started,forced_calls=calls)
                (HERE/'metadata_screen_rows_local.json').write_text(json.dumps(record,indent=2)+'\n')
                assert len(record['rows'])==108,lines[-1]
                assert {r['cell'] for r in record['rows']}=={'route','pool_epilogue','tail','both'}
                assert {r['case'] for r in record['rows']}==set(range(12))
                assert not any('INCOMPLETE' in line for line in lines),lines
                assert all(r['rounds']==6 and len(r['samples'][0])==6 for r in record['rows'])
                return lines
            module._lab_run_one=run_one
        return module
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_moe_w4a16/lab/lab_best7_route_verify_glue.py'),
        '--no-baseline','--tests-only','--json',str(HERE/'metadata_screen_local.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:assert len(record.get('rows',[]))==108,'lab error hidden by host success'
        raise


if __name__=='__main__':main()
