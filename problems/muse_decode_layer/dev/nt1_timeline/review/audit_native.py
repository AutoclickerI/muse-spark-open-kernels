"""Count full sm90a native instructions/resources for each exact independent arm."""
import collections,hashlib,json,re,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[4];rows={};m=json.loads((D/'manifest.json').read_text())
if len(sys.argv)>1 and (D/'native_audit.json').exists():rows=json.loads((D/'native_audit.json').read_text())
for arm,spec in m.items():
 if len(sys.argv)>1 and arm not in sys.argv[1:]:continue
 cu=R/'runs/screen/p3a_review'/f'{arm.lower()}.cu';obj=cu.with_suffix('.o');log=cu.with_suffix('.compile.log')
 r=subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_90a','-c',str(cu),'-o',str(obj),'-Xptxas=-v'],capture_output=True,text=True);log.write_text(r.stdout+r.stderr);r.check_returncode()
 sass=subprocess.check_output(['cuobjdump','--dump-sass',str(obj)],text=True);cu.with_suffix('.sass').write_text(sass)
 name,body=next((n,b) for n,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S) if 'moe_varILi1ELi69E' in n)
 op=collections.Counter(k.split('.')[0] for k in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]*)',body))
 matches=[tuple(map(int,x[1:])) for x in re.findall(r'Function properties for (\S*moe_varILi1ELi69E\S*)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers',r.stderr)]
 assert len(matches)==1 and op.get('HMMA')==32 and op.get('HMUL2')==128,(arm,matches,op)
 stack,stores,loads,registers=matches[0]
 rows[arm]=dict(cuda_sha256=spec['cuda_sha256'],full_flags69_opcodes=dict(op),registers=registers,stack_bytes=stack,spill_store_bytes=stores,spill_load_bytes=loads)
 (D/'native_audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(arm,registers,stack,stores,loads,dict(op),flush=True)
