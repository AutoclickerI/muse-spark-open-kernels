from pathlib import Path
import sys,json,time,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(HERE.parent)]
import local_eval,verify_best
inject='--inject-flip' in sys.argv
record={};original=local_eval.load_module

def loader(name,path):
    module=original(name,path)
    if name=='submission':
        def run_one(label,source,deadline,weights,state):
            import linecache
            filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
            ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
            calls={}
            def install(m):calls.update(verify_best.install(m,False))
            ns['CG_LOCAL_INSTALL']=install
            if inject:
                original_run=ns['fg_run']
                def wrong_route(m,data,state,plan,cell='all'):
                    got=original_run(m,data,state,plan,cell)
                    if cell=='pool' and data['config']['tokens']==1:
                        ids=got[2].clone();ids[0,7]=(ids[0,7]+1)%256
                        return got[0],got[1],ids
                    return got
                ns['fg_run']=wrong_route
            started=time.monotonic();lines=ns['lab'](deadline,weights,state)
            record.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=ns['CG_ROWS'],disabled=ns['CG_DISABLED'],lines=lines,
                elapsed_s=time.monotonic()-started,forced_calls=calls)
            (HERE/('flip_tolerance_local.json' if inject else 'screen_rows_local.json')).write_text(json.dumps(record,indent=2)+'\n')
            assert not any('INCOMPLETE' in x for x in lines),lines[-1]
            if inject:
                assert len(record['disabled'])==1,record['disabled']
                assert all(x['cell']=='pool' and x['reason']=='route_flip' for x in record['disabled'])
                assert len(record['rows'])==174,len(record['rows'])
            assert len({r['case'] for r in record['rows']}|{r['case'] for r in record['disabled']})==12
            assert {r['cell'] for r in record['rows']}|{r['cell'] for r in record['disabled']}=={'pool','tail','all','epi','route'}
            print('\n'.join(lines),flush=True)
            return lines
        module._lab_run_one=run_one
    return module
local_eval.load_module=loader
sys.argv=[sys.argv[0],str(ROOT/'problems/muse_bf16_proj/lab/lab_best7_coop_glue.py'),'--no-baseline','--tests-only','--json',str(HERE/'screen_local.json')]
try:local_eval.main()
except SystemExit as e:
    if e.code==0:assert record,'lab error hidden by host success'
    raise
