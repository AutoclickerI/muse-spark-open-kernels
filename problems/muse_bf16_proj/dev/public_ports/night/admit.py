"""Admit a fixed code template with only constrained parent-selection constants mutable."""
import ast,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
MUTABLE={'NG_PARENT_ID','NG_PARENT_POINTS','NG_PARENT_POOL'}
def code_hash(text):
 tree=ast.parse(text)
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(getattr(t,'id','') in MUTABLE for t in n.targets):n.value=ast.Constant(value='<validated-parent-selection>')
 return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
def admit():
 p=json.loads((H/'local_validation.json').read_text());s=json.loads((H/'stress_validation.json').read_text());n=json.loads((H/'native_validation.json').read_text());assert all(x['passed'] for x in (p,s,n));assert json.loads((H/'candidate_validation.json').read_text())['passed'];assert json.loads((H/'pipeline_validation.json').read_text())['passed'];assert p['layer_TASK']==s['layer_TASK']==512 and p['strict_TASK']==s['strict_TASK']==208
 manifest=json.loads((H/'manifest.json').read_text());source=(H/'out/source.py').read_bytes();assert hashlib.sha256(source).hexdigest()==manifest['source_sha256']
 fixed=code_hash((H/'out/night_gap.py').read_text());assert p['code_template_sha256']==s['code_template_sha256']==fixed;assert p['payload_sha256']==manifest['payload_sha256']
 out=dict(passed=True,code_template_sha256=fixed,source_sha256=manifest['source_sha256'],S21_git_blob=subprocess.check_output(['git','rev-parse','bf1eeda:problems/muse_decode_layer/cand_r3_best7_s21_final.py']).decode().strip(),body_sha256=manifest['body_sha256'],mutable_constants=sorted(MUTABLE),parent_membership='Must be a subset of exact frozen whole_manifest PORTS points and pool arms; validated before dispatch.',proofs=dict(primary=p,stress=s,native=n),limits='Local glue and sm90a compile admission, not native performance. Parent receipt must be complete and exact before a next h200 send; no polling or quota bypass.')
 (H/'admission.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(admit(),indent=2))
