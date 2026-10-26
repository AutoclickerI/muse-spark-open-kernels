"""Read-only provenance proof; never import or execute either untrusted clone."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;review=json.loads((H/'review.json').read_text());a=argparse.ArgumentParser();a.add_argument('--clones',type=Path,default=Path(review['clone_root']));a=a.parse_args();out={}
for name,repo,file in [('operatorstack','operatorstack','muse_attention_decode/submission.py'),('lee101','lee101','muse_attention_decode/submission_v56.py')]:
 p=a.clones/repo;s=(p/file).read_text();sha=hashlib.sha256(s.encode()).hexdigest();pin=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip();assert pin==review[name]['commit'];assert sha==review[name]['source_sha256'];out[name]=dict(commit=pin,source_sha256=sha)
 if name=='operatorstack':
  old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};new={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((H/'operatorstack.py.inc').read_text()).body if isinstance(n,ast.FunctionDef)}
  unchanged=[n for n in review[name]['selected_functions'] if n!='_fused'];assert all(old[n]==new[n] for n in unchanged);out[name]['identical_kernel_AST']=unchanged;out[name]['changed_host']='_fused removes unselected pair path; _hopper explicit local forcing; owned setup/attention ABI'
 else:
  values=[n.value.value for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)]
  cu=max(values,key=len);lines=cu.splitlines(True);selected=''.join(''.join(lines[lo-1:hi]) for lo,hi in review[name]['cuda_ranges']);copy=(H/'lee_kernels.cu.inc').read_text();assert all(''.join(lines[lo-1:hi]) in copy for lo,hi in review[name]['cuda_ranges']),'Kernel ranges differ';out[name]['identical_cuda_ranges']=review[name]['cuda_ranges'];out[name]['changed_host']='Only selected launch families, original flags and smem layout, S19 tensor ABI, launch-error checks'
 notices=(H/'NOTICES.txt').read_text();assert (p/'LICENSE').read_text().strip() in notices
 if (p/'NOTICE').exists():assert (p/'NOTICE').read_text().strip() in notices
out['pass']=True;(H/'provenance.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
