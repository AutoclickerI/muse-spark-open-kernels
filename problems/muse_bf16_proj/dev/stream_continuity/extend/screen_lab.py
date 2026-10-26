"""Run every frozen arm/seed/shape through the public checker and actual captures locally."""
import importlib.util,json,lzma,pathlib,sys,time
import torch
H=pathlib.Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import local_eval
sp=importlib.util.spec_from_file_location('ew_s18screen',R/'problems/muse_decode_layer/dev/s18/screen.py');screen=importlib.util.module_from_spec(sp);sp.loader.exec_module(screen)
forced='--forced' in sys.argv;mode='forced' if forced else 'normal'
ns=dict(SC_LOCAL_INSTALL=lambda m:screen.install(m,forced))
src=(H/'lab_generated.py.inc').read_text();exec(compile(src,str(H/'lab_generated.py.inc'),'exec'),ns)
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
old=ns['ew_pack']
def retain(result):
    (H/'out'/('raw_'+mode+'.json.xz')).write_bytes(lzma.compress(json.dumps(result).encode()))
    return old(result)
ns['ew_pack']=retain
ns['EW_LOCAL_PROGRESS']=lambda cases:print('SHAPE',cases[-1]['case'],'TASK',cases[-1]['failures'],flush=True)
lines=ns['lab'](time.monotonic()+2400,None,None);(H/'out'/('receipt_'+mode+'.txt')).write_text('\n'.join(lines)+'\n')
x=json.loads(lzma.decompress((H/'out'/('raw_'+mode+'.json.xz')).read_bytes()));assert x['TASK_checks']==3456 and not any(c['failures'] for c in x['cases'])
print('PASS',mode,x['TASK_checks'],x['elapsed_s'],'s',sum(len(l)+1 for l in lines),'chars',flush=True)
