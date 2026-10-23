"""Build a register-row-base gather copy from the accepted pipeline."""
from pathlib import Path
import ast
root=Path(__file__).resolve().parents[1]
source=(root/'cand_xqa_pipeline.py').read_text(); tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='X_CUDA' for t in n.targets))
c=ast.literal_eval(node.value)
c=c.replace('int rows[KT/64];','uint64_t rows[KT/64];',1)
c=c.replace('rows[rr]=(tileid<ntiles&&key<end)?pt[key>>p.lg]*p.ps+(key&(p.ps-1)):-1;', 'rows[rr]=(tileid<ntiles&&key<end)?(uint64_t)(pt[key>>p.lg]*p.ps+(key&(p.ps-1)))*rowbytes+h*64:UINT64_MAX;',1)
old='''int physical=rows[rr];bool valid=physical>=0;physical=valid?physical:0;
   cp(sk+slot*KB+off(row,col),p.kc+physical*rowbytes+h*64+col,valid);
   cp(sv+slot*KB+off(row,col),p.vc+physical*rowbytes+h*64+col,valid);'''
new='''uint64_t base=rows[rr];bool valid=base!=UINT64_MAX;
   cp(sk+slot*KB+off(row,col),valid?p.kc+base+col:p.kc,valid);
   cp(sv+slot*KB+off(row,col),valid?p.vc+base+col:p.vc,valid);'''
if old not in c: raise SystemExit('issue pattern not found')
c=c.replace(old,new,1)
lines=source.splitlines(True); lines[node.lineno-1:node.end_lineno]=['X_CUDA = r"""\n',c,'\n"""\n']
out=''.join(lines).replace('muse_xqa_pipeline_sm','muse_xqa_gather2_sm')
(root/'cand_xqa_gather2.py').write_text(out);(root/'dev/xqa_gather2.cu').write_text(c);print('Built',root/'cand_xqa_gather2.py')
