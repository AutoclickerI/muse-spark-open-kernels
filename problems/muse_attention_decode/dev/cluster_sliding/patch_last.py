"""Build the last-CTA sliding candidate on frozen S11. Experimental until native paired evidence lands."""
import ast,hashlib,re,sys
from pathlib import Path
import patch_cluster,build_last
HERE=Path(__file__).resolve().parent

def patch(src,rows=((1,1,True),),nk=128,ns=None):
    out=patch_cluster.patch(src,rows,nk,ns)
    d=(HERE/'last_driver.py').read_text();cell='\n'.join(ast.get_source_segment(d,n) for n in ast.parse(d).body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ','_COUNTERS') for t in n.targets)))
    for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','_COUNTERS','module','plan','attention'):cell=re.sub(r'\b'+name+r'\b','cs_'+name,cell)
    cuda=build_last.source();cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
    a=out.index('import hashlib\ncs_CUDA=');b=out.index('\n_CLUSTER_ROWS=',a)
    out=out[:a]+'import hashlib\ncs_CUDA='+repr(cuda)+'\ncs_CPP='+repr(cpp)+'\n'+cell+out[b:]
    return out

if __name__=='__main__':
    out=patch(Path(sys.argv[1]).read_text());p=Path(sys.argv[2]);p.write_text(out);print(p,hashlib.sha256(out.encode()).hexdigest())
