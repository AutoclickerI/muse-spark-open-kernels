"""Screen exact import wrapper and all continuity arms with S17 local stand-ins."""
import importlib.util,json,lzma,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import local_eval
sp=importlib.util.spec_from_file_location('sc_s17_screen',R/'problems/muse_decode_layer/dev/s17/screen.py');screen=importlib.util.module_from_spec(sp);sp.loader.exec_module(screen)
forced='--forced' in sys.argv;original=local_eval.load_module;lines=[]
raw_path=H/'out'/('forced_raw.json.xz' if forced else 'normal_raw.json.xz');raw_path.unlink(missing_ok=True)
def loader(name,path):
    if name!='submission':return original(name,path)
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    def install():
        screen.install(m,forced);m.SC_LOCAL_INSTALL=lambda v:screen.install(v,forced)
        run=m._lab_run_one
        def checked(name,source,deadline,weights,state):
            # Preserve actual full payload, retain its result before packing.
            import linecache
            fname='<sc-local-payload>';linecache.cache[fname]=(len(source),None,source.splitlines(True),fname)
            ns=dict(m.__dict__);exec(compile(source,fname,'exec'),ns)
            pack=ns['sc_pack']
            def retain(result):
                (H/'out'/('forced_raw.json.xz' if forced else 'normal_raw.json.xz')).write_bytes(lzma.compress(json.dumps(result).encode()))
                return pack(result)
            ns['sc_pack']=retain
            out=ns['lab'](time.monotonic()+1800,weights,state);lines.extend(out);return out
        m._lab_run_one=checked
    m._SC_LOCAL_INSTALL=install
    source=path.read_text();marker='\n_lab_import_start = ';assert source.count(marker)==1
    exec(compile(source.replace(marker,'\n_SC_LOCAL_INSTALL()'+marker,1),str(path),'exec'),m.__dict__)
    return m
local_eval.load_module=loader
mode='forced' if forced else 'normal'
sys.argv=['local_eval.py',str(H/'out/wrapper.py'),'--no-baseline','--reps','4','--inputs','3','--json',str(H/('screen_'+mode+'.json'))]
try:local_eval.main()
finally:
    (H/'out'/('receipt_'+mode+'.txt')).write_text('\n'.join(lines)+'\n')
    raw=H/'out'/('forced_raw.json.xz' if forced else 'normal_raw.json.xz')
    assert raw.exists(), 'Full payload did not complete; inspect LAB_ERROR in screen log'
    if raw.exists():
        result=json.loads(lzma.decompress(raw.read_bytes()));bad={c['case']:c['failures'] for c in result['cases'] if c['failures']}
        print('PAYLOAD_TASK',result['task_checks'],'FAILURES',bad,flush=True)
        assert result['task_checks']==1728 and len(result['cases'])==12 and not bad
