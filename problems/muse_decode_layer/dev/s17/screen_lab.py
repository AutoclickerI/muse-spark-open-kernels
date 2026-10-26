"""Exercise the exact S17 import wrapper and every off/alternative arm locally."""
import importlib.util,json,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parent/'s16')]
import local_eval
sp=importlib.util.spec_from_file_location('s17_screen',H/'screen.py');screen=importlib.util.module_from_spec(sp);sp.loader.exec_module(screen)
sp=importlib.util.spec_from_file_location('s17_report',H/'report.py');report=importlib.util.module_from_spec(sp);sp.loader.exec_module(report)
forced='--forced' in sys.argv;late='--late' in sys.argv;original=local_eval.load_module;lines=[]
manifest=json.loads((H/'generated'/('lab_late_manifest.json' if late else 'lab_manifest.json')).read_text())


def loader(name,path):
    if name!='submission':return original(name,path)
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    def install():
        screen.install(m,forced);m.S16_LOCAL_INSTALL=lambda v:screen.install(v,forced)
        run=m._lab_run_one
        def checked(name,source,deadline,weights,state):
            out=run(name,source,time.monotonic()+1800,weights,state);lines.extend(out);return out
        m._lab_run_one=checked
    m._S17_LOCAL_INSTALL=install
    source=path.read_text();marker='\n_lab_import_start = ';assert source.count(marker)==1
    exec(compile(source.replace(marker,'\n_S17_LOCAL_INSTALL()'+marker,1),str(path),'exec'),m.__dict__)
    return m


local_eval.load_module=loader
mode=('late_' if late else '')+('forced' if forced else 'normal');out=H/'generated'/('lab_'+mode+'_local.json')
sys.argv=['local_eval.py',str(R/manifest['wrapper']),'--tests-only','--json',str(out)]
try:local_eval.main()
except SystemExit as exc:
    if exc.code==0:
        import base64,lzma
        blob=[l.split(' ',1)[1] for l in lines if l.startswith('S17_XZ85 ')];assert len(blob)==1,lines[-3:]
        result=report.decode(json.loads(lzma.decompress(base64.b85decode(blob[0]))))
        result['local_wrapper_sha256']=manifest['wrapper_sha256']
        (H/'generated'/('lab_'+mode+'_validation.json')).write_text(json.dumps(result,indent=2)+'\n')
        assert result['complete'] and not result['failures'],result['failures']
    raise
