"""Official local layer screen and exact packed-lab rehearsal with Hopper stand-ins."""
import importlib.util,sys,time,hashlib,json
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
MERGE_SCREEN='--merge' in sys.argv
if MERGE_SCREEN:sys.argv.remove('--merge')
s19=load('da_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');portable=load('da_portable',H/'portable.py')
def install(m,forced=False):
    records=s19.install(m,forced)
    m.DA_DO_CODE=True
    m.DA_DO_MERGE=MERGE_SCREEN
    if hasattr(m,'DA_SELECTED'):
        m.DA_LOCAL_REDUCE=True;m.DA_WIDE_STANDIN=True;m.DA_LOCAL_MM=portable.mm
    return records
if '--lab' in sys.argv:
    p=load('da_whole_payload',H/'whole_payload.py');p.WHOLE_LOCAL_INSTALL=lambda m:install(m,True);p.WHOLE_LOCAL_ARCHIVE=lambda d:(H/'whole_raw_local.json').write_text(json.dumps(d))
    if '--quick' in sys.argv:p.WHOLE_LOCAL_CASES={(1,1,2048),(16,1,2048),(32,1,0)}
    buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum();result=p.lab(time.monotonic()+3600,None);print('\n'.join(result));d=json.loads((H/'whole_raw_local.json').read_text());assert d['failure'] is None,d['failure'];assert d['complete'] or '--quick' in sys.argv
    (H/'whole_lab_validation.json').write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'whole_payload.py').read_bytes()).hexdigest(),rows=len(d['rows']),checks=d['checks']),indent=2)+'\n')
else:
    # Reuse the exact S19 local_eval entry, adding only the new native stand-ins.
    s19.old.install=install;s19.old.main()
