"""Screen the complete second-round wrapper and all off-arm graphs locally."""
import importlib.util,json,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval,screen,static_cache
forced='--forced' in sys.argv
original=local_eval.load_module;lines=[]


def loader(name,path):
    if name!='submission':return original(name,path)
    # Import-phase labs execute before importlib returns. Install local Hopper
    # stand-ins at that boundary, before synthetic prepare and graph capture.
    import types
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    def before_import():
        screen.install(m,forced);m.S16_LOCAL_INSTALL=lambda v:screen.install(v,forced)
        run=m._lab_run_one
        def checked(name,source,deadline,weights,state):
            out=run(name,source,time.monotonic()+1800,weights,state);lines.extend(out);return out
        m._lab_run_one=checked
    m._S16_BEFORE_IMPORT=before_import
    source=path.read_text();marker='\n_lab_import_start = '
    assert source.count(marker)==1
    source=source.replace(marker,'\n_S16_BEFORE_IMPORT()'+marker,1)
    exec(compile(source,str(path),'exec'),m.__dict__)
    return m


local_eval.load_module=loader
mode='forced' if forced else 'normal';out=H/'generated'/('lab_'+mode+'_local.json')
manifest=json.loads((H/'generated/lab_manifest.json').read_text())
sys.argv=['local_eval.py',str(R/manifest['wrapper']),'--tests-only','--json',str(out)]
try:local_eval.main()
except SystemExit as exc:
    if exc.code==0:
        import base64,lzma
        blobs=[line.split(' ',1)[1] for line in lines if line.startswith('S16_XZ85 ')]
        assert len(blobs)==1,lines[-3:]
        result=json.loads(lzma.decompress(base64.b85decode(blobs[0])))
        (H/'generated'/('lab_'+mode+'_validation.json')).write_text(json.dumps(result,indent=2)+'\n')
        assert result['complete'] and not result['failures'],result['failures']
    raise
