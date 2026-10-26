import concurrent.futures,hashlib,json,re,subprocess,sys
from pathlib import Path
import kernels
H=Path(__file__).resolve().parent

def one(item):
 name,s=item;p=H/'out'/(name+'.cu');p.write_text(s)
 r=subprocess.run(['nvcc','-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-cubin',str(p),'-o',str(p.with_suffix('.cubin'))],capture_output=True,text=True)
 log=r.stdout+r.stderr;p.with_suffix('.log').write_text(log)
 if r.returncode:raise RuntimeError((name,log[-4500:]))
 resource=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.cubin'))],text=True)
 print('PASS',name,flush=True)
 return dict(name=name,sha256=hashlib.sha256(s.encode()).hexdigest(),resources=resource,compile_warnings=[x for x in log.splitlines() if 'warning' in x])
if __name__=='__main__':
 items=[(name+'_'+k,s) for name,v in kernels.sources().items() for k,s in (v.items() if isinstance(v,dict) else [(name,v)])]
 only=sys.argv[1:]
 if only:items=[x for x in items if x[0] in only]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,items))
 if only:
  prior=json.loads((H/'native.json').read_text());changed={x['name'] for x in rows};rows=[x for x in prior if x['name'] not in changed]+rows
 (H/'native.json').write_text(json.dumps(rows,indent=2)+'\n')
