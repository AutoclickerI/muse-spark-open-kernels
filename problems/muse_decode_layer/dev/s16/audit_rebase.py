"""CPU-only Python audit and proof that row narrowing preserves native code."""
import ast, copy, json, sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval, patches


def native_strings(source):
    return sorted(n.value for n in ast.walk(ast.parse(source))
                  if isinstance(n,ast.Constant) and isinstance(n.value,str)
                  and ('__global__' in n.value or '#include' in n.value))


def main():
    failures={};proof=[];roundtrips=[]
    for name,spec in patches.registry().items():
        full,meta=patches.compose([spec]);narrow=copy.deepcopy(spec);narrow['rows']=spec['rows'][:1]
        source,_=patches.compose([narrow])
        assert native_strings(full)==native_strings(source),name
        restored,restored_meta=patches.compose(json.loads(json.dumps([spec])))
        assert restored==full and restored_meta['sha256']==meta['sha256'],name
        roundtrips.append(dict(arm=name,sha256=meta['sha256'],selection_json_roundtrip_exact=True))
        path=H/'generated/audit_sources'/f'{name}.py';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(full)
        checks=local_eval.undefined_names(path)+local_eval.call_signatures(path)
        if not all(row['ok'] for row in checks):failures[name]=checks
        proof.append(dict(arm=name,cuda_identical_after_row_specialization=True,full_sha256=meta['sha256']))
    (H/'specialization_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    (H/'selection_roundtrip.json').write_text(json.dumps(dict(passed=True,arms=roundtrips),indent=2)+'\n')
    result=dict(passed=not failures,arms=len(proof),failures=failures)
    (H/'python_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    assert not failures,failures


if __name__=='__main__':main()
