"""Hopper code/resource audit; split phase kernels retain full math,combine has none."""
import collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[4];m=json.loads((D/'cells.json').read_text());rows={}
for name,spec in m.items():
 cu=R/'runs/screen/p3a_structural'/(name.lower()+'.cu');obj=cu.with_suffix('.o');log=cu.with_suffix('.compile.log')
 p=subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_90a','-c',str(cu),'-o',str(obj),'-Xptxas=-v'],capture_output=True,text=True);log.write_text(p.stdout+p.stderr);p.check_returncode();sass=subprocess.check_output(['cuobjdump','--dump-sass',str(obj)],text=True);cu.with_suffix('.sass').write_text(sass);parts={}
 for fn,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S):
  v=re.search(r'moe_varILi1ELi69E(?:Li([0-3])E)?',fn)
  if not v:continue
  mode=int(v[1] or 0);ops=collections.Counter(k.split('.')[0] for k in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]*)',b))
  rr=re.search(r'Function properties for '+re.escape(fn)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers',p.stderr);assert rr
  stack,stores,loads,regs=map(int,rr.groups());assert not stores and not loads,(name,mode,rr.groups())
  if mode<3:assert ops['HMMA']==32 and ops['HMUL2']==128,(name,mode,ops)
  else:assert not ops['HMMA'] and not ops['HMUL2'],(name,mode,ops)
  parts[str(mode)]=dict(registers=regs,stack_bytes=stack,spill_stores=stores,spill_loads=loads,ops=dict(ops))
 if spec['split']:assert set(parts)=={'0','1','2','3'}
 rows[name]=dict(cuda_sha256=spec['cuda_sha256'],modes=parts);print(name,{k:(v['registers'],v['ops'].get('HMMA',0)) for k,v in parts.items()},flush=True)
 (D/'native_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
