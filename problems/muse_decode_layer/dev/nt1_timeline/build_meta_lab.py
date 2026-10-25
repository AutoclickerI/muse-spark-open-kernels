"""Freeze exact S11 control, cluster metadata producer, consumer and separated pairing driver."""
import ast,hashlib
from pathlib import Path
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
def build():
    src=(ROOT/'prof/lab/moe_nt1_reuse.py').read_text()
    tree=ast.parse((ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text())
    fields={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and getattr(n.targets[0],'id','') in ('GLUE_CL_CPP','GLUE_CL_CUDA')}
    src+='\nMETA_GLUE_CPP='+repr(fields['GLUE_CL_CPP'])+'\nMETA_GLUE_CUDA='+repr(fields['GLUE_CL_CUDA'])+'\n'
    src+=(DEV/'meta_cell.py').read_text()+'\n'+(DEV/'meta_lab.py.inc').read_text()
    out=ROOT/'prof/lab/moe_nt1_meta.py';out.write_text(src);ast.parse(src)
    print(out.relative_to(ROOT),hashlib.sha256(src.encode()).hexdigest())
if __name__=='__main__':build()
