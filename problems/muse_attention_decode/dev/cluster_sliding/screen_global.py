"""Run exact wrapper lab and production host through local_eval/audit, including forced host glue."""
import ast,importlib.util,linecache,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'tools'))
import local_eval
original=local_eval.load_module

def loader(name,path):
    m=original(name,path)
    if name!='submission':return m
    import torch
    if torch.cuda.get_device_capability()[0]!=9:
        m._ap_fa3_scheduler_metadata=lambda *a,**kw:None
        for n in ast.parse((ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text()).body:
            if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':m.VQ8_CUDA=ast.literal_eval(n.value)
        print('LOCAL ONLY: N128 persistent ring uses explicit frozen N64 launch stand-in; exact N128 tile math checked separately. FA3 scheduler metadata stand-in None on sm120.',flush=True)
    def run(name,src,deadline,weights,state):
        fn='<local_p5_global_attention_lab>';linecache.cache[fn]=(len(src),None,src.splitlines(True),fn);ns=dict(m.__dict__)
        exec(compile(src,fn,'exec'),ns);ns['CS_SEEDS']=ns['CS_SEEDS'][:2];ns['CS_ROUNDS']=2
        lines=ns['lab'](deadline,weights,state)
        assert len(ns['CS_RESULTS'])==4
        assert all(len(p['nodes'])==1 for r in ns['CS_RESULTS'] for n,p in r['profiles'].items() if n in ('C64','C128','W4','W8','OLD') and (n!='OLD' or r['case'].endswith('S'))), 'capture needs exactly one attention launch'
        return lines
    m._lab_run_one=run
    return m
local_eval.load_module=loader
spec=importlib.util.spec_from_file_location('p5_audit',ROOT/'problems/muse_decode_layer/dev/s9/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);audit.main()
