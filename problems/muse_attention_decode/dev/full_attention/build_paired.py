"""Adjacent one-head CTAs gather the two halves of each physical KV row together."""
import ast
import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'onehead.cu').read_text()
def sub(a,b,n=1):
 global s
 assert s.count(a)==n,(a,s.count(a),n)
 s=s.replace(a,b)
sub('(int)HK, (int)(B * HK)','(int)(HK/2), (int)(B*(HK/2))')
sub('(int)min_tiles, (int)grid,','(int)min_tiles, (int)(grid/2),')
sub('(int)(grid / p.NS)','(int)(p.grid / p.NS)')
sub('cfg.gridDim = dim3(p.grid);','cfg.gridDim = dim3(2*p.grid);')
sub('cta = blockIdx.x;','cta = blockIdx.x>>1;')
sub('i = blockIdx.x;','i = blockIdx.x>>1;')
sub('if ((int)blockIdx.x >= S.G)','if ((int)(blockIdx.x>>1) >= S.G)')
sub('if (i == 0)  // the combine','if (blockIdx.x == 0)  // the combine')
sub('(int64_t)hp * 64 + c * 16','(int64_t)(2*hp+(blockIdx.x&1)) * 64 + c * 16')
sub('const int kvh = hp;','const int kvh = 2*hp+(blockIdx.x&1);')
sub('((hp) * 8 + (row & 7))','((2*hp+(blockIdx.x&1)) * 8 + (row & 7))')
# Each paired CTA writes one disjoint 64-row half of a shared logical slot.
sub('* 64 * PR','* 128 * PR',3)
sub('slot * 64 +','slot * 128 + (blockIdx.x&1)*64 +')
sub('pb * 64 +','pb * 128 +')
sub('const int head = (hp) * 8 + (row & 7);','const int head = (2*hp+hh) * 8 + (row & 7);')
sub('dim3(p.L * p.wpr, p.NS)','dim3(2*p.L * p.wpr, p.NS)')
sub('if (HQ != 8 * HK','if (grid%2 || HK%2 || HQ != 8 * HK')
(H/'paired.cu').write_text(s)
tree=ast.parse((H/'onehead.py').read_text())
for n in tree.body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':n.value=ast.Constant(s)
 if isinstance(n,ast.FunctionDef) and n.name=='vq8_attention':
  for q in n.body:
   if isinstance(q,ast.Assign) and getattr(q.targets[0],'id','')=='NS':q.value=ast.parse('B*(HK//2)',mode='eval').body
drv=ast.unparse(tree).replace('(G + NS) * 64 * 68','(G//2 + NS) * 128 * 68')+'\n'
(H/'paired.py').write_text(drv)
test=(H/'check_onehead.py').read_text().replace("load('onehead')","load('paired')").replace("'onehead.py'","'paired.py'").replace("'onehead_validation.json'","'paired_validation.json'")
(H/'check_paired.py').write_text(test)
prefix=(H/'phases_lab.py').read_text().split('\nSOURCES=',1)[0]
lab=(H/'hybrid_lab.py.inc').read_text()
lab=lab.replace('range(5)','range(8)').replace('range(3)','range(6)')
lab=lab.replace("for st in (4,6,8)]","for name in ('onehead','paired') for st in (4,6,8)]").replace("mods['new']['vq8_attention']","mods[name]['vq8_attention']").replace('lambda st=st:','lambda st=st,name=name:')
lab=lab.replace("for j,st in enumerate((4,6,8))","for j,st in enumerate(('one4','one6','one8','pair4','pair6','pair8'))")
lab=lab.replace('FP8 QK + register FP16 PV','one-head CTA160 native register-V PV, independent/adjacent head scheduling')
lab=lab.replace('V stays raw, exact register widening; P keeps E4M3 quantization, output register transpose only once per portion.','Three CTA occupancy target; adjacent physical CTAs share page/tile ranges and read complementary 64B row halves; raw V, shared P.')
(H/'occupancy_lab.py').write_text(prefix+'\nSOURCES='+repr({'old':(H/'vq_runtime.py').read_text(),'onehead':(H/'onehead.py').read_text(),'paired':drv})+'\n'+lab)
print('Paired heads:',hashlib.sha256(s.encode()).hexdigest())
