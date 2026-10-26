"""Full packed-payload rehearsal using P4's local Hopper stand-ins."""
import importlib.util,sys,types,time,json,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3]
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
p=module('da_lab_local',H/'payload.py');screen=module('da_s18_screen',R/'problems/muse_decode_layer/dev/s18/screen.py');old=p.da_load
portable=module('da_portable',H/'portable.py')
def load():
    ns=old();m=types.ModuleType('dense_all_local');m.__dict__.update(ns);screen.install(m,False);ns.update(vars(m));prepare=ns['da_prepare'];ns['da_prepare']=lambda weights:prepare(weights,True,True);ns['DA_LOCAL_MM']=portable.mm
    class VD:
        def __getitem__(self,grid):
            def run(x,w,p,xd,wd,t,n,k,s,*a,**kw):
                assert tuple(p.shape)==(s,t,n) and kw.get('launch_pdl')
                p.zero_();portable.mm(x,w,p[0])
            return run
    ns['_vdsk_gemm']=VD();return ns
p.da_load=load;p.DA_LOCAL_TRACE=True;p.DA_LOCAL_ARCHIVE=lambda data:(H/'lab_raw_local.json').write_text(json.dumps(data))
debug='--pool-debug' in sys.argv or '--pair-debug' in sys.argv
if debug:p.DA_LOCAL_CASES={(128,'pool_router')};p.DA_LOCAL_SYNC=True
if '--pair-debug' in sys.argv:p.DA_LOCAL_CASES.add((128,'o'))
if '--sync-debug' in sys.argv:p.DA_LOCAL_SYNC=True
buf=torch.empty(1<<28,device='cuda',dtype=torch.uint8);p.lab_cold_l2=lambda:buf.view(torch.int64).sum();begin=time.monotonic();lines=p.lab(begin+3600,None);print('\n'.join(lines),flush=True);assert lines[-1].startswith('END complete='+str(len(p.DA_LOCAL_CASES))+' ' if debug else 'END complete=36');assert sum(len(x)+1 for x in lines)<=9000
(H/('pool_debug_validation.json' if debug else 'lab_validation.json')).write_text(json.dumps(dict(passed=True,payload_sha256=hashlib.sha256((H/'payload.py').read_bytes()).hexdigest(),elapsed_s=time.monotonic()-begin,lines=lines),indent=2)+'\n')
