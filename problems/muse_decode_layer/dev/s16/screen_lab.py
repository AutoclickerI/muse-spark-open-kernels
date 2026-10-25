"""Screen the complete second-round wrapper and all off-arm graphs locally."""
import importlib.util,json,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval,screen,static_cache
forced='--forced' in sys.argv
original=local_eval.load_module;lines=[]


def loader(name,path):
    m=original(name,path)
    if name=='submission':
        screen.install(m,forced);m.S16_LOCAL_INSTALL=lambda v:screen.install(v,forced)
        run=m._lab_run_one
        def checked(name,source,deadline,weights,state):
            out=run(name,source,time.monotonic()+1800,weights,state);lines.extend(out);return out
        m._lab_run_one=checked
    return m


local_eval.load_module=loader
mode='forced' if forced else 'normal';out=H/'generated'/('lab_'+mode+'_local.json')
sys.argv=['local_eval.py',str(R/'problems/muse_decode_layer/lab/lab_s16_payload.py'),'--tests-only','--json',str(out)]
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
