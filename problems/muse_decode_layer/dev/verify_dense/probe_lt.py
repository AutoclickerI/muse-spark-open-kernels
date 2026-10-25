from pathlib import Path
import importlib.util,time,torch,json
h=Path(__file__).resolve().parent;sp=importlib.util.spec_from_file_location('vd',h/'projection.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
w=torch.randn((2304,8192),device='cuda',dtype=torch.bfloat16);x=torch.randn((128,8192),device='cuda',dtype=torch.bfloat16)
s=time.monotonic();op=m.VDLt(w,128);print('ENUM',op.count,op.heuristics,'s',time.monotonic()-s,flush=True)
ref=x.float()@w.float().T;rows=[]
for index in (0,op.heuristics,op.count-1):
 out=op(x,index);torch.cuda.synchronize();row=dict(index=index,error=float((out.float()-ref).norm()/ref.norm()),selection=op.selection(index));rows.append(row);print(row,flush=True)
(h/'lt_probe.json').write_text(json.dumps(dict(count=op.count,heuristics=op.heuristics,seconds=time.monotonic()-s,rows=rows),indent=2)+'\n')
