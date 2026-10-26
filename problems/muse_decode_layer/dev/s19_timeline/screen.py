"""Exercise the exact import payload locally, with explicit S19 Hopper stand-ins."""
import importlib.util,json,lzma,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import torch
from local_eval import wrapper_smoke
from lab_packet import encode_packet
sp=importlib.util.spec_from_file_location('s19_screen',H.parent/'s19/screen.py')
screen=importlib.util.module_from_spec(sp);sp.loader.exec_module(screen)
ns={'__name__':'s19_timeline_local'}
exec(compile((H/'generated/body.py').read_text(),str(H/'generated/body.py'),'exec'),ns)
def install(m):
    screen.install(m,forced=True)
    # Screen counters hold original functions in keyword defaults; remove those observers
    # before namespace cloning so RT_ARM remains local to each capture's globals.
    for name,fn in list(vars(m).items()):
        kw=getattr(fn,'__kwdefaults__',None) or {}
        if '_fn' in kw:setattr(m,name,kw['_fn'])
ns['LOCAL_INSTALL']=install
ns['LOCAL_RETAIN']=lambda x:(H/'local_raw.json.xz').write_bytes(lzma.compress(json.dumps(x).encode(),preset=9))
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8)
ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
print(wrapper_smoke('lab-only'),flush=True)
result=ns['lab'](time.monotonic()+1500,None,None)
wire=encode_packet('s19_timeline','local',result['packet'])
assert len(wire.encode())+1<=11744,('wire size',len(wire))
(H/'local_receipt.json').write_text(json.dumps(result)+'\n')
print('PASS local forced;TASK',result['packet']['checks'],'wire',len(wire),flush=True)
