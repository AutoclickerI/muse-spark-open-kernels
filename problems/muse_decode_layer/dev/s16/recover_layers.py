"""Repair only static entry points in the three queued P4 layer audits."""
import ast,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
SOURCES={'nearby':('5d0893419ae8828070b8705bff53d3b905b709ee13e171dfb4505e8835c5e5db','lab_nearby_payload.py'),
         'release':('f067eb263e1a67ac5277cec91b032fa33de4891ec3a12fdba9a24b23ae3797bc','lab_release_payload.py'),
         'library':('54aa3bedc460bbc8880892042a76993872a384a308fb90b5e007fd97b7fe8b2c','lab_library_payload.py')}


def main():
    result={}
    for name,(digest,filename) in SOURCES.items():
        src=(R/'runs/sent'/digest[:16]/filename).read_text();assert hashlib.sha256(src.encode()).hexdigest()==digest
        assert not any(isinstance(n,ast.FunctionDef) and n.name=='kernel_fn' for n in ast.parse(src).body)
        src+='\n_p4_entry_kernel=globals()["kernel_fn"]\ndef kernel_fn(data,state):\n    return _p4_entry_kernel(data,state)\n'
        path=R/'problems/muse_decode_layer/lab'/('lab_s15_'+name+'_repaired.py');path.write_text(src)
        result[name]=dict(original_sha256=digest,sha256=hashlib.sha256(src.encode()).hexdigest(),file=str(path.relative_to(R)),
                          repair='explicit kernel_fn definition only; exact original payload and freeze criteria',screened=False)
    (H/'layer_recovery.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
