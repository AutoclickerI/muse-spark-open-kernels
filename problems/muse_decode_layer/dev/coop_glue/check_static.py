from pathlib import Path
import sys,json,tempfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import local_eval
source=(ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text();result={}
for relative in ('coop_glue/best7_coop_glue_cell.py','qkv_overlap/best7_kv_group_cell.py'):
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'cell.py';p.write_text(source+'\n'+(HERE.parent/relative).read_text())
        checks=dict(undefined=local_eval.undefined_names(p),signatures=local_eval.call_signatures(p))
        assert all(row['ok'] for errors in checks.values() for row in errors),(relative,checks)
        result[relative]=checks
(HERE/'cell_static.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
