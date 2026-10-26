"""Full paired wrapper with real TASK gates and local Hopper launch stand-ins."""
import importlib.util,sys,time,json,base64,lzma,ast
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools')]
import local_eval
def load_module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
stand=load_module('p10_pool_lab_stand',R/'problems/muse_decode_layer/dev/s15/local_standins.py');forced='--forced' in sys.argv;loader=local_eval.load_module;records=[]
def load(name,path):
    delayed=None
    if name=='submission':
        source=Path(path).read_text();tree=ast.parse(source)
        node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and getattr(n.value.func,'id','')=='_lab_run')
        delayed=ast.get_source_segment(source,node);lines=source.splitlines(True);lines[node.lineno-1:node.end_lineno]=['pass # import lab delayed for local Hopper stand-ins\n']
        stage=H/'screen_layer_lab'/'delayed.py';stage.parent.mkdir(exist_ok=True);stage.write_text(''.join(lines));m=loader(name,stage)
    else:m=loader(name,path)
    if name=='submission':
        if forced:load_module('p10_pool_lab_force',R/'problems/muse_decode_layer/dev/best7/force.py').install(m,False,False)
        stand.install(m)
        if forced:
            original_vq=m._vq8_prepare
            def trace_vq(*a,**kw):
                result=original_vq(*a,**kw);print('P10_VQ_ADMISSION_TRACE',repr(result),repr(m._VQ8_ROWS),repr(original_vq.__globals__.get('_VQ8_ROWS')),flush=True);return result
            m._vq8_prepare=trace_vq
        run=m._lab_run_one
        def invoke(name,source,deadline,weights,state):
            lines=run(name,source,time.monotonic()+2400,weights,state);records.extend(lines);return lines
        m._lab_run_one=invoke
        exec(compile(delayed,'<p10-local-import-lab>','exec'),m.__dict__)
    return m
local_eval.load_module=load
out=H/'screen_layer_lab'/('forced' if forced else 'normal');out.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'pool_layer_host.py'),'--tests-only','--json',str(out/'result.json')]
try:local_eval.main()
finally:
    (out/'lab.json').write_text(json.dumps(dict(lines=records,complete=bool(records and records[-1]=='END complete=12/12 checks=768')),indent=2)+'\n')
