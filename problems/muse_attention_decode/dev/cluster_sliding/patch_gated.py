"""Single attention launch including gate, onto frozen S11. Experimental until paired whole-layer promotion."""
import ast,hashlib,sys
from pathlib import Path
import patch_cluster,build_gated

def patch(src,nk=128,late=False):
    out=patch_cluster.patch(src,nk=nk)
    lines=out.splitlines(True)
    n=next(n for n in ast.parse(out).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='cs_CUDA')
    lines[n.lineno-1:n.end_lineno]=['cs_CUDA='+repr(build_gated.source(late))+'\n'];out=''.join(lines)
    old="def _gate_split_on(data, state, plan):\n    cfg = data['config']\n"
    assert out.count(old)==1
    out=out.replace(old,old+"    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in _CLUSTER_ROWS:return False\n")
    return out

if __name__=='__main__':
    out=patch(Path(sys.argv[1]).read_text());p=Path(sys.argv[2]);p.write_text(out);print(p,hashlib.sha256(out.encode()).hexdigest())
