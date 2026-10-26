"""Run the actual host-state payload locally; cold replays and all arms/seeds intact."""
import importlib.util,json,lzma,pathlib,sys,time
import torch
H=pathlib.Path(__file__).resolve().parent;R=H.parents[3];P=R/'problems/muse_decode_layer';sys.path[:0]=[str(P),str(R/'tools')]
import local_eval
m=local_eval.load_module('p7_tail_host',P/'cand_r3_best7_s15.py');stand=local_eval.load_module('p7_tail_host_stand',P/'dev/s15/local_standins.py');stand.install(m)
ref=local_eval.load_module('reference',P/'reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
weights=ref.generate_weights(ref.WEIGHTS_SEED);state=m.prepare(weights);ns=dict(m.__dict__);ns['__name__']='p7_tail_local_lab'
exec(compile((H/'lab_generated.py.inc').read_text(),str(H/'lab_generated.py.inc'),'exec'),ns)
# Fresh generator uses the same local scheduler stand-in as the official local screen.
ns['kernel_fn'].__globals__['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
old=ns['mt_pack']
def retain(result):
 (H/'local_raw.json.xz').write_bytes(lzma.compress(json.dumps(result).encode(),preset=9));return old(result)
ns['mt_pack']=retain
lines=ns['lab'](time.monotonic()+800,weights,state);(H/'local_receipt.txt').write_text('LOCAL SM120\n'+'\n'.join(lines)+'\n');print('PASS lab',sum(len(l)+1 for l in lines),'chars',flush=True)
