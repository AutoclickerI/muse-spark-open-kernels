"""Run the exact batch in the proj host, with explicit large-smem stand-ins."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import local_eval
original=local_eval.load_module;records={}

def load(name,path):
    module=original(name,path)
    if name=='submission':
        def run(label,source,deadline,weights,state):
            import linecache
            import torch
            fname='<h200_lab '+label+'>';linecache.cache[fname]=(len(source),None,source.splitlines(True),fname)
            ns=dict(module.__dict__);exec(compile(source,fname,'exec'),ns)
            # Hopper has228KiB shared/SM; this local Blackwell GPU has100KiB.
            # Exercise the same Python branches/partial ABI using a shorter
            # native ring. Native4/8-stage ordering is independently tested.
            standins={}
            for c,(r,kc,st,ks,cta) in list(ns['BR_CONFIGS'].items()):
                if r*kc*st*2+st*8>torch.cuda.get_device_properties(0).shared_memory_per_block_optin:
                    ns['BR_CONFIGS'][c]=(r,kc,st//2,ks,cta);standins[c]=[st,st//2]
            cls=ns['BulkRingProjection']
            prop=torch.cuda.get_device_properties(0);real_library=ns['br_library']()
            reservations=[]
            class CudaProxy:
                def __getattr__(self,key):return getattr(torch.cuda,key)
                def get_device_properties(self,device):
                    class HopperProperties:
                        major=9
                        multi_processor_count=132
                    return HopperProperties()
            class TorchProxy:
                cuda=CudaProxy()
                def __getattr__(self,key):return getattr(torch,key)
            class LibraryStandin:
                def br_run(self,*args):
                    v=list(args);requested=v[13];v[13]=min(requested,prop.shared_memory_per_block_optin)
                    if v[12]==0:reservations.append([requested,v[13]])
                    return real_library.br_run(*v)
            # Execute the actual Hopper residency/prepare branch. Its driver
            # stand-in caps dynamic smem to this GPU's100KiB, while the shorter
            # ring retains the original shape, packing and partial ownership.
            ns['torch']=TorchProxy();library=LibraryStandin();ns['br_library']=lambda:library
            class ForcedGrid(cls):
                def __init__(self,*a,**kw):
                    config=a[2];kw['grid']=132*ns['BR_CONFIGS'][config][-1];super().__init__(*a,**kw)
            ns['BulkRingProjection']=ForcedGrid
            started=time.monotonic();lines=ns['lab'](min(deadline,started+180.),weights,state)
            rows=ns['BR_ROWS'];records.update(rows=rows,lines=lines,elapsed_s=time.monotonic()-started,standins=standins,
                hopper_reservations=sorted(set(map(tuple,reservations))))
            (HERE/'rows_local.json').write_text(json.dumps(records,indent=2)+'\n')
            assert len([r for r in rows if r['kind']=='role'])==16
            assert len([r for r in rows if r['kind']=='segment'])==4
            assert all(len(r['trials'])==21 for r in rows if r['kind']=='role')
            assert all(r['diagnostic']['units_max']-r['diagnostic']['units_min']<=1 for r in rows if r['kind']=='role')
            assert sum(len(x)+1 for x in lines)<=8500
            assert not any('ERROR' in x or 'INCOMPLETE' in x for x in lines)
            assert any(r[0]==122880 for r in reservations) and any(r[0]==81920 for r in reservations)
            records['admitted']=True;return lines
        module._lab_run_one=run
    return module

local_eval.load_module=load
host=ROOT/'problems/muse_bf16_proj/lab/lab_dense_bulk_ring.py'
sys.argv=[sys.argv[0],str(host),'--tests-only','--no-baseline','--json',str(HERE/'host_local.json')]
try:local_eval.main()
except SystemExit as exc:
    if exc.code==0:
        assert records.get('admitted'),'payload did not complete'
        report=dict(payload_sha256=hashlib.sha256((ROOT/'prof/lab/dense_bulk_ring.py').read_bytes()).hexdigest(),
            host_sha256=hashlib.sha256(host.read_bytes()).hexdigest(),roles=16,segments=4,configs=21,host_tests=9,
            elapsed_s=records['elapsed_s'],forced_hopper_grid=True,large_smem_standins=records['standins'],
            forced_hopper_prepare=True,reservation_standins=records['hopper_reservations'],
            real_host_weights=True,log_characters=sum(len(x)+1 for x in records['lines']))
        (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('BR_HOST_ADMISSION',json.dumps(report),flush=True)
    raise
