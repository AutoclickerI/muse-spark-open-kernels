"""Assemble a self-contained K128/K256 TMA-ring lab, without changing sent files."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def build():
    tail=(HERE/'lab_tma_tail.py').read_text()
    tail=tail.replace('vp_tma_module','vg_tma_module').replace('vp_tma_choices','vg_tma_choices')
    tail=tail.replace('vp_tma(','vg_tma(').replace('vp_tma_lab','vg_tma_lab')
    tail=tail.replace('bt,ns,ncw,st,mc,ks,waves,pipe=c','bt,ns,ncw,st,mc,ks,waves,pipe,kg=c')
    tail=tail.replace('mc,ks,groups,pipe)','mc,ks,groups,pipe,kg)')
    tail=tail.replace("family=configs[index][:5]",
        "family=(configs[index][0],configs[index][1],configs[index][2],configs[index][4],configs[index][-1])")
    tail=tail.replace('VP_TMA','VG_TMA').replace('VP_CHAIN','VG_CHAIN')
    tail=tail.replace('waves,async)', 'waves,async,KG)')
    tail=tail.replace('(64,1,1,4,1,1,0,1),(64,1,1,4,2,2,1,1)',
                      '(64,1,1,3,1,1,0,1,2),(32,1,1,2,2,2,1,1,4)')
    tail=tail.replace('dev/verify/validate.py','dev/verify/validate_grouped.py')
    path=ROOT/'prof/lab/verify_proj_tma_v2.py'
    path.write_text((HERE/'grouped_projection.py').read_text()+'\n'+
                    (HERE/'lab_library_tail.py').read_text().split('\ndef lab(')[0]+'\n'+tail)
    return path


if __name__=='__main__':print(build())
