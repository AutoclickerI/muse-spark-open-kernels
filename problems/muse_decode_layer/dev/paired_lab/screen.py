"""Run the real lab wrapper and host official checks, with Hopper stand-ins."""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev/best6')]
import local_eval
import force as best6_force


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--share-s',type=float,default=124.);args=ap.parse_args()
    load=local_eval.load_module;records={}
    def loader(name,path):
        module=load(name,path)
        if name=='submission':
            def run_one(label,source,deadline,weights,state):
                import linecache
                filename='<h200_lab '+label+'>'
                linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
                calls={}
                def install(m):calls.update(best6_force.install(m,False))
                ns['DP_LOCAL_INSTALL']=install
                started=time.monotonic()
                lines=ns['lab'](min(deadline,started+args.share_s),weights,state)
                rows=ns['DP_ROWS'];records.update(rows=rows,lines=lines,elapsed_s=time.monotonic()-started,
                    forced_calls=calls,share_s=args.share_s)
                assert len({(r['case'],r['seed']) for r in rows if r['stage']=='G'})==36,lines[-1]
                assert len([r for r in rows if r['stage']=='F'])==108,lines[-1]
                assert len([r for r in rows if r['stage']=='E'])==72,lines[-1]
                assert not any('REJECT' in line or 'INCOMPLETE' in line for line in lines),lines
                assert all(r['rounds']==6 and len(r['samples'][0])==6 for r in rows)
                assert sum(len(line)+1 for line in lines)<=4500
                (HERE/'paired_rows_local.json').write_text(json.dumps(records,indent=2)+'\n')
                return lines
            module._lab_run_one=run_one
        return module
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/lab/lab_decode_paired.py'),
              '--no-baseline','--json',str(HERE/'screen_local.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:assert len(records.get('rows',[]))==216,'lab failed despite the host passing'
        raise


if __name__=='__main__':main()
