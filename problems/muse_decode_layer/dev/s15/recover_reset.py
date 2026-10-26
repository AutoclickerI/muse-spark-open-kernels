"""Repair the rejected wrapper's static entry point; all measured code is exact."""
import ast,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
OLD='49ce378bb4bbb287731b16a0ab6ac99f5b165c65697d7d98b69438f5c0aa1260'


def main():
    source=(R/'runs/sent'/OLD[:16]/'lab_s15ab_payload.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==OLD
    assert not any(isinstance(n,ast.FunctionDef) and n.name=='kernel_fn' for n in ast.parse(source).body)
    source+='\n_s15_entry_kernel=globals()["kernel_fn"]\ndef kernel_fn(data,state):\n    return _s15_entry_kernel(data,state)\n'
    path=R/'problems/muse_decode_layer/lab/lab_s15ab_payload.py';path.write_text(source)
    q=json.loads((H/'queue.json').read_text());q.update(rejected_previous_wrapper_sha256=OLD,
        rejection='422 no kernel_fn function defined; no job created',wrapper_sha256=hashlib.sha256(source.encode()).hexdigest(),
        status='repaired static entry; local screens required before immediate priority -1 check',job=None)
    (H/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
    print(path,q['wrapper_sha256'])


if __name__=='__main__':main()
