from pathlib import Path
import sys,json,time,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import local_eval
original=local_eval.load_module;record={}

def loader(name,path):
    module=original(name,path)
    if name=='submission':
        def one(label,source,deadline,weights,state):
            import linecache
            filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
            ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
            def install():
                cls=ns['VDCutlass']
                def call(self,x):
                    self._call(x,False)
                    ns['torch'].mm(x,self.w.T,out=self.out,out_dtype=ns['torch'].float32) if self.fp32 else ns['torch'].mm(x,self.w.T,out=self.out)
                    return self.out
                cls.__call__=call
            ns['VD_LOCAL_INSTALL']=install
            start=time.monotonic();lines=ns['lab'](deadline,weights,state)
            record.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=ns['VD_ROWS'],disabled=ns['VD_DISABLED'],lines=lines,elapsed_s=time.monotonic()-start)
            (HERE/'screen_rows_local.json').write_text(json.dumps(record,indent=2)+'\n')
            assert not any('INCOMPLETE' in line for line in lines),lines[-1]
            assert {(r['m'],r['role']) for r in record['rows']}=={(m,role) for m in (64,128,256) for role in ('qkv_gate','o','pool_router','unpooler')}
            assert {r['cell'] for r in record['rows']} >= {'lt_search','deep_gemm','cutlass0','cutlass1','cutlass2'}
            print('\n'.join(lines),flush=True);return lines
        module._lab_run_one=one
    return module
local_eval.load_module=loader
sys.argv=[sys.argv[0],str(ROOT/'problems/muse_moe_w4a16/lab/lab_verify_dense_library.py'),'--no-baseline','--tests-only','--json',str(HERE/'screen_local.json')]
try:local_eval.main()
except SystemExit as e:
    if e.code==0:assert record,'lab exception hidden by host pass'
    raise
