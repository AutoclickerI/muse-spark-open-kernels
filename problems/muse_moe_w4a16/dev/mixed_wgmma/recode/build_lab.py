"""Embed matched raw/recoded WGMMA, stock NT1-recoded and masked Humming."""
import ast,hashlib,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3];sys.path.insert(0,str(P))
from build_prefill_lab import subset

def main():
 original=(R/'prof/lab/moe_w2fix_m256.py').read_text()
 control=subset(original,('rg_generate_weights','dc_fixtures','dc_oracles','wx_control_prepare','pp_reference','MWGraph'))
 engine=subset((P/'engine.py').read_text(),('mw_repack_planes','mw_route'))
 text=(H/'recoded.py').read_text();tree=ast.parse(text)
 tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in ('engine','nt1_pack'))]
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='RC_CUDA' for a in n.targets):n.value=ast.Constant((H/'recode.cu').read_text()+(H/'nt1.cu').read_text())
  if isinstance(n,ast.FunctionDef) and n.name=='rc_module':
   n.body.insert(0,ast.parse("wx_need(40.,'native compile')").body[0])
 native=ast.unparse(ast.fix_missing_locations(tree));out=R/'prof/lab/moe_recoded_m256.py'
 out.write_text(control+'\n'+engine+'\n'+(H/'nt1_pack.py').read_text()+'\n'+native+'\n'+(H/'driver.py.inc').read_text())
 print(out.relative_to(R),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)

if __name__=='__main__':main()
