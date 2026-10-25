"""Raw V -> native A fragments via LDSM transpose, warp shuffles and byte permutation."""
import ast
import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
drivers={}
for parent,name in [('regv','ldsv'),('onehead','ldsvone'),('paired','ldsvpair')]:
 s=(H/(parent+'.cu')).read_text()
 a=s.index('DI void pv(');b=s.index('DI void restore_output',a)
 pv=s[a:b]
 assert pv.count('col=32*kk+16*hf+2*t;')==1
 pv=pv.replace('col=32*kk+16*hf+2*t;','col=32*kk+16*hf+4*t;').replace('ps+off(row,col+8)','ps+off(row,col+2)')
 x=pv.index('#pragma unroll\n for(int kk=0;kk<2;++kk)',pv.index('float* al='));y=pv.index(' wg_fence();',x)
 pv=pv[:x]+''' // LDSM.T covers32 keys x16 features per warp. Permute each16-key
 // block into (2t,2t+1,8+2t,9+2t), matching shared P's key permutation.
#pragma unroll
 for(int kk=0;kk<2;++kk){
  uint32_t z0,z1,z2,z3;
  ldsm4t(z0,z1,z2,z3,su(v+off(32*kk+8*(lane>>3)+(lane&7),16*w)));
#pragma unroll
  for(int rh=0;rh<2;++rh){
   int src=4*((g+8*rh)/2)+t;
   uint32_t b0=__shfl_sync(0xffffffffu,z0,src),b1=__shfl_sync(0xffffffffu,z1,src);
   uint32_t b2=__shfl_sync(0xffffffffu,z2,src),b3=__shfl_sync(0xffffffffu,z3,src);
   a[4*kk+rh]=prmt(b0,b1,(g&1)?0x7531:0x6420);
   a[4*kk+2+rh]=prmt(b2,b3,(g&1)?0x7531:0x6420);
  }
 }
'''+pv[y:]
 s=s[:a]+pv+s[b:]
 (H/(name+'.cu')).write_text(s)
 tree=ast.parse((H/(parent+'.py')).read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(s)
 drv=ast.unparse(tree)+'\n';drivers[name]=drv
 (H/(name+'.py')).write_text(drv)
 test=(H/('check_'+parent+'.py')).read_text().replace("load('"+parent+"')","load('"+name+"')").replace("'"+parent+".py'","'"+name+".py'").replace("'"+parent+"_validation.json'","'"+name+"_validation.json'")
 (H/('check_'+name+'.py')).write_text(test)
 print(name,hashlib.sha256(s.encode()).hexdigest())
prefix=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
lab=(H/'hybrid_lab.py.inc').read_text()
lab=lab.replace('range(5)','range(11)').replace('range(3)','range(9)')
lab=lab.replace("for st in (4,6,8)]","for name in ('ldsv','ldsvone','ldsvpair') for st in (4,6,8)]").replace("mods['new']['vq8_attention']","mods[name]['vq8_attention']").replace('lambda st=st:','lambda st=st,name=name:')
lab=lab.replace("for j,st in enumerate((4,6,8))","for j,st in enumerate(('two4','two6','two8','one4','one6','one8','pair4','pair6','pair8'))")
lab=lab.replace('FP8 QK + register FP16 PV','native FP8 raw-V register-A via LDSM.T, two/one/paired-head occupancy')
lab=lab.replace('V stays raw, exact register widening; P keeps E4M3 quantization, output register transpose only once per portion.','Raw V stays untransposed in shared; two LDSM.T + warp shuffles/PRMT form A; shared P uses matching key permutation; stage/occupancy sweep.')
(H/'ldsv_lab.py').write_text(prefix+'\nSOURCES='+repr({'old':(H/'vq_runtime.py').read_text(),**drivers})+'\n'+lab)
