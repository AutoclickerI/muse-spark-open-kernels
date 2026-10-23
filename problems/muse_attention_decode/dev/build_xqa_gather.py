"""Build a private gather optimized copy from the accepted pipeline source."""
from pathlib import Path
import ast

root = Path(__file__).resolve().parents[1]
source = (root / 'cand_xqa_pipeline.py').read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'X_CUDA' for t in n.targets))
c = ast.literal_eval(node.value)
old = r'''int rows[KT/64];
 auto load_rows=[&](int tileid){
  #pragma unroll
  for(int rr=0;rr<KT/64;rr++){
   int key=begin+tileid*KT+tid/4+rr*64;
   rows[rr]=(tileid<ntiles&&key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;
  }
 };
 auto issue=[&](int tileid,int slot){
  #pragma unroll
  for(int rr=0;rr<KT/64;rr++){
   int row=tid/4+rr*64,col=(tid&3)*16;
   int physical=rows[rr];bool valid=physical>=0;physical=valid?physical:0;
   cp(sk+slot*KB+off(row,col),p.kc+physical*rowbytes+h*64+col,valid);
   cp(sv+slot*KB+off(row,col),p.vc+physical*rowbytes+h*64+col,valid);
  }
 };'''
new = r'''uint64_t* rowbase=reinterpret_cast<uint64_t*>(vh+2*KB);
 auto load_rows=[&](int tileid,int slot){
  // Store one byte base per logical row; K and V reuse it below.
  if(p.ps==1 && tid<32){
   int key=begin+tileid*KT+tid*4;
   if(tileid<ntiles && key+3<end && (key&3)==0){
    int4 x=*reinterpret_cast<const int4*>(pt+key);
    rowbase[slot*KT+tid*4+0]=(uint64_t)x.x*rowbytes+h*64;
    rowbase[slot*KT+tid*4+1]=(uint64_t)x.y*rowbytes+h*64;
    rowbase[slot*KT+tid*4+2]=(uint64_t)x.z*rowbytes+h*64;
    rowbase[slot*KT+tid*4+3]=(uint64_t)x.w*rowbytes+h*64;
   }else for(int i=0;i<4;i++){int k=key+i;rowbase[slot*KT+tid*4+i]=(tileid<ntiles&&k<end)?(uint64_t)pt[k]*rowbytes+h*64:UINT64_MAX;}
  }else if(tid<KT){
   int k=begin+tileid*KT+tid;
   rowbase[slot*KT+tid]=(tileid<ntiles&&k<end)?(uint64_t)(p.ps==1?pt[k]:pt[k>>p.lg]*p.ps+(k&(p.ps-1)))*rowbytes+h*64:UINT64_MAX;
  }
  __syncthreads();
 };
 auto issue=[&](int tileid,int slot){
  #pragma unroll
  for(int rr=0;rr<KT/64;rr++){
   int row=tid/4+rr*64,col=(tid&3)*16;
   uint64_t base=rowbase[slot*KT+row];bool valid=base!=UINT64_MAX;
   cp(sk+slot*KB+off(row,col),valid?p.kc+base+col:p.kc,valid);
   cp(sv+slot*KB+off(row,col),valid?p.vc+base+col:p.vc,valid);
  }
 };'''
if old not in c:
    raise SystemExit('gather pattern not found')
c = c.replace(old, new, 1)
c = c.replace('load_rows(stage);issue(stage,stage);', 'load_rows(stage,stage);issue(stage,stage);', 1)
c = c.replace('load_rows(ST-1);', 'load_rows(ST-1,ST-1);', 1)
c = c.replace('load_rows(i+ST);', 'load_rows(i+ST,(i+ST)%ST);', 1)
c = c.replace('constexpr int smem=ST*2*KB+8192+(P8?0:2*KB);', 'constexpr int smem=ST*2*KB+8192+(P8?0:2*KB)+ST*KT*8;', 1)
lines = source.splitlines(True)
lines[node.lineno-1:node.end_lineno] = ['X_CUDA = r"""\n', c, '\n"""\n']
out = ''.join(lines).replace('muse_xqa_pipeline_sm', 'muse_xqa_gather_sm')
(root / 'cand_xqa_gather.py').write_text(out)
(root / 'dev/xqa_gather.cu').write_text(c)
print('Built', root / 'cand_xqa_gather.py')
