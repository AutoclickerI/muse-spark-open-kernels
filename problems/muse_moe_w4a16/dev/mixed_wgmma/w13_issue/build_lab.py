"""One MoE-host matched W13 issue/loader ablation with the proven raw M256 cell."""
import ast,sys,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3];sys.path.insert(0,str(P))
from build_prefill_lab import subset

def main():
 source=(R/'prof/lab/moe_w2fix_m256.py').read_text();controls=subset(source,('dc_weights','dc_fixtures','dc_oracles','wx_control_prepare','pp_reference','MWGraph'))
 helpers=subset((P/'recode/recoded.py').read_text(),('rc_repack','rc_masks','RC_SIGNS','RC_ZEROS'))
 htree=ast.parse(helpers);htree.body=[n for n in htree.body if not (isinstance(n,ast.ImportFrom) and n.module in ('engine','nt1_pack')) and not (isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='RC_CUDA' for a in n.targets))];helpers=ast.unparse(ast.fix_missing_locations(htree))
 text=(H/'adapter.py').read_text();t=ast.parse(text);t.body=[n for n in t.body if not (isinstance(n,ast.ImportFrom) and n.module in ('recoded','engine'))]
 for n in t.body:
  if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='FX_CUDA' for a in n.targets):n.value=ast.Constant((H/'base.cu').read_text()+(H/'issue.cu').read_text()+(P/'recode/recode.cu').read_text())
  if isinstance(n,ast.FunctionDef) and n.name=='fx_module':n.body.insert(0,ast.parse("wx_need(40.,'native compile')").body[0])
 engine=subset((P/'engine.py').read_text(),('mw_repack_planes','mw_route'))
 out=R/'prof/lab/moe_w13_issue_m256.py';out.write_text(controls+'\n'+engine+'\n'+helpers+'\n'+ast.unparse(ast.fix_missing_locations(t))+'\n'+(H/'driver.py.inc').read_text())
 print(out.relative_to(R),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)

if __name__=='__main__':main()
