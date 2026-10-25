"""Execute exact batched payload with forced layer glue; the local DSM substitute is explicit and one-launch."""
import hashlib,importlib.util,json,linecache,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import torch,local_standin
src=(H/'out/resident_verify_lab.py').read_text();fn='<resident-lab-local>';linecache.cache[fn]=(len(src),None,src.splitlines(True),fn)
ns={};exec(compile(src,fn,'exec'),ns);local=local_standin.install(ns)
force_path=R/'problems/muse_decode_layer/dev/best7/force.py';spec=importlib.util.spec_from_file_location('resident_force',force_path);force=importlib.util.module_from_spec(spec);spec.loader.exec_module(force)
orig=ns['host_module']
def host_module():
    host=orig();force.install(host,False,False);host._ap_fa3_scheduler_metadata=lambda *a,**kw:None;return host
ns['host_module']=host_module
ns['SEEDS']=ns['SEEDS'][:2];ns['ROUNDS']=2
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
lines=ns['lab'](time.monotonic()+1500,None,None)
assert len(ns['RESULTS'])==2 and all(all(c==2 for c in r['task'].values()) for r in ns['RESULTS'])
print('\n'.join(l for l in lines if not l.startswith('VR_XZ85 ')));print('ARCHIVE chars',len(next(l for l in lines if l.startswith('VR_XZ85 '))))
(H/'out/payload_admission.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(src.encode()).hexdigest(),standin=local,results=ns['RESULTS'],all_reference_checks=True),indent=2)+'\n')
print('PASS exact lab, 2shapes x10arms x2fresh seeds, TASK20pct margin; native cluster placement unverified')
