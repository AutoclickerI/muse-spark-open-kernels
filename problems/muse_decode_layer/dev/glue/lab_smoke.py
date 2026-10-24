from pathlib import Path
import importlib.util,time,torch,json
ROOT=Path(__file__).resolve().parents[4]
p=ROOT/'prof/lab/decode_glue.py';sp=importlib.util.spec_from_file_location('glue_lab',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=m.lab(time.monotonic()+240,{'device':torch.empty(1,device='cuda')})
print('\n'.join(rows),flush=True)
(ROOT/'problems/muse_decode_layer/dev/glue/lab_valid_local.json').write_text(json.dumps(rows,indent=2)+'\n')
