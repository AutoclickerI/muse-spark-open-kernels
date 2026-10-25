"""Freeze register-A WGMMA source for local screens, SASS audit and the one-file lab."""
from pathlib import Path
H=Path(__file__).resolve().parent
src=(H/'wg.py').read_text().replace('WG_KERNEL = None', 'WG_KERNEL = '+repr((H/'wg_kernel.cu').read_text()))
(H/'wg_frozen.py').write_text(src)
# No local imports remain in the judge payload.
body=(H/'coded.py').read_text()+'\n'+src.replace('from coded import CD_CUDA, cd_encode, cd_decode\n','')
(H/'wg_body.py').write_text(body)
print(H/'wg_frozen.py')

pre=(H/'lab_body.py').read_text().split('\ndef lab(',1)[0].replace('__CL_VS__','(3,)').replace('__CL_SHAPES__','((8,1,2),)')
(H/'wg_lab.py').write_text('import triton\nimport triton.language as tl\n'+body+'\n'+pre+'\n'+(H/'wg_lab.py.inc').read_text())
