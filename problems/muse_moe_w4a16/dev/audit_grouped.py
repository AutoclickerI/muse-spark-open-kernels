"""Inspect the actually assembled offline SM90a object, without claiming GPU execution."""
from pathlib import Path
import re
import subprocess
p=Path(__file__).resolve().parent
obj=Path('/data2/jihoon/muse_kernel/img-home/.cache/torch_extensions/py312_cu130/moe_round2_90a_e4e09da77db7/cuda.cuda.o')
sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(obj)],text=True)
(p/'grouped_sm90.sass').write_text(sass)
for section in sass.split('Function : ')[1:]:
 name=section.splitlines()[0]
 m=re.search(r'groupedILi(\d+)ELi(\d+)ELi(\d+)',name)
 if not m or m[2]!='1' or m[3]!='2':continue
 metrics={x:len(re.findall(r'\b'+x+r'(?:\.|\b)',section)) for x in ('HGMMA','HMMA','LDGSTS','LDL','STL','WARPGROUP','BAR')}
 print('SM90a W13 N='+m[1],metrics)
