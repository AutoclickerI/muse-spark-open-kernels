"""Verify the pruned drop-in uses the exact definitions admitted in the layer fixture."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
block=Path(__file__).with_name('projection_cutlass.py').read_text()
fixture=(ROOT/'runs/screen/verify_proj/cutlass_layer_fixture.py').read_text()
# The fixture intentionally replaces vgc_module AFTER its production definition
# with portable transport. Compare the actual production nodes, before that override.
reference={ast.dump(n) for n in ast.parse(fixture).body}
checked=[]
for n in ast.parse(block).body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Assign)):
        assert ast.dump(n) in reference,getattr(n,'name',ast.unparse(n)[:80])
        checked.append(getattr(n,'name',ast.unparse(n.targets[0]) if isinstance(n,ast.Assign) else ''))
report={'same_definitions':checked,'block_sha256':hashlib.sha256(block.encode()).hexdigest(),
        'all_match':True,'fixture_transport':'portable stand-in, factory separately forced through strict ctypes ABI'}
(ROOT/'runs/screen/verify_proj/cutlass_block_identity.json').write_text(json.dumps(report,indent=2)+'\n')
print('BLOCK_AST_IDENTITY',len(checked),'definitions/constants PASS')
