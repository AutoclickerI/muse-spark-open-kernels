from pathlib import Path
import re
import subprocess
p=Path(__file__).resolve().parent
obj=Path('/data2/jihoon/muse_kernel/img-home/.cache/torch_extensions/py312_cu130/moe_round2_90a_8e029da70967/cuda.cuda.o')
s=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(obj)],text=True)
(p/'tma_sm90.sass').write_text(s)
for block in s.split('Function : ')[1:]:
 name=block.splitlines()[0]
 if not re.search(r'groupedILi128ELi[01]ELi2ELi2',name):continue
 print(name,{k:len(re.findall(r'\b'+k+r'(?:\.|\b)',block)) for k in ('HGMMA','HMMA','UBLKCP','LDGSTS','LDL','STL','WARPGROUP','BAR')})
