"""Check a frozen best7 file, so concurrent table edits cannot relabel evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval
source=ROOT/'problems/muse_decode_layer/cand_r3_best7.py'
body=source.read_bytes();sha=hashlib.sha256(body).hexdigest()
snapshot=ROOT/'runs/screen'/f'p7_best7_{sha[:16]}.py';snapshot.write_bytes(body)
spec=importlib.util.spec_from_file_location('p7_force_driver',ROOT/'problems/muse_decode_layer/dev/best7/force.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
load=local_eval.load_module;record={}
def loader(name,path):
    m=load(name,path)
    if name=='submission':record['forced']=driver.install(m,False,True,True)
    return m
local_eval.load_module=loader
sys.argv=[sys.argv[0],str(snapshot),'--no-baseline','--json',str(HERE/'best7_frozen_local.json')]
try:local_eval.main()
except SystemExit as exc:
    if exc.code==0:
        d=json.loads((HERE/'best7_frozen_local.json').read_text());assert d['sha256']==sha
        assert record['forced']['prefetch_ops']==6 and record['forced']['stream_ops']==24
        for k,count in (('tests',13),('shapes',12),('static',16)):
            assert len(d[k])==count and all(r['ok'] for r in d[k]),k
        record.update(sha256=sha,snapshot=str(snapshot.relative_to(ROOT)))
        (HERE/'best7_frozen_dispatch_local.json').write_text(json.dumps(record,indent=2)+'\n')
    raise
