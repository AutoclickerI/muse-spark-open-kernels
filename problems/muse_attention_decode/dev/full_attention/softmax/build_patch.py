"""Generate a selected softmax-only patch onto exact S14 final; selection requires H200 evidence."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3]
ap=argparse.ArgumentParser();ap.add_argument('variant',choices=['mix25','mix50','gap','mixgap','tc25','ping','ping25']);a=ap.parse_args()
p=R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py';old=p.read_text()
assert hashlib.sha256(old.encode()).hexdigest().startswith('f1c77ce6')
cu=(H/(a.variant+'.cu')).read_text();tag=hashlib.sha256(cu.encode()).hexdigest()[:12]
lines=old.splitlines(True);edits=[]
for n in ast.parse(old).body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':edits.append((n.lineno-1,n.end_lineno,'VQ8_CUDA = r"""'+cu+'"""\n'))
 if isinstance(n,ast.FunctionDef) and n.name=='_vq8_module':edits.append((n.lineno-1,n.end_lineno,''.join(lines[n.lineno-1:n.end_lineno]).replace('muse_vq8_c262e4c5f9c3_',f'muse_vq8_{tag}_')))
assert len(edits)==2
for lo,hi,text in sorted(edits,reverse=True):lines[lo:hi]=[text]
new=''.join(lines);compile(new,'s14_softmax.py','exec')
out=H/('s14_'+a.variant+'.py');out.write_text(new)
patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+str(p.relative_to(R)),tofile='b/'+str(p.relative_to(R))))
(H/('s14_'+a.variant+'.patch')).write_text(patch)
proof=dict(base_sha256=hashlib.sha256(old.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),variant=a.variant,promotion=False)
(H/('s14_'+a.variant+'_identity.json')).write_text(json.dumps(proof,indent=2)+'\n')
print(proof)
