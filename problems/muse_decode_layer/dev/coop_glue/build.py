from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
old=HERE.parent/'fused_glue'
s=(old/'runtime.py.inc').read_text()
s=s.replace('self.t=8 if recipe==0 else 16','self.t=8')
s=s.replace('self.lids=torch.empty_like(self.ids)','self.lids=torch.empty_like(self.ids)\n        self.aux=torch.empty(tokens*18,device=dev,dtype=torch.float32)')
s=s.replace('self.wts,self.lids,self.counts]','self.wts,self.lids,self.counts,self.aux]')
s=s.replace('(_fgct.c_void_p*20)','(_fgct.c_void_p*21)')
s=s.replace('if tokens<=8 else', 'if tokens<=8 else')
(HERE/'runtime.py.inc').write_text(s)
source='FG_CUDA='+repr((HERE/'kernel.cu').read_text())+'\n'+s
(HERE/'dropin.py').write_text(source)
print(hashlib.sha256(source.encode()).hexdigest())
