"""Replace the measured 2.11-us top8 selection; freeze the projection body."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
src=(HERE/'stats_parallel.py').read_text();tree=ast.parse(src)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ps_topk')
old=ast.get_source_segment(src,node)
start=old.index('    keys=')
new=old[:start]+'''    ids,wts=warp_top8(ordered,scores,e)
    tl.store(IDS+t*8+e,ids,e<8);tl.store(WTS+t*8+e,wts,e<8)
    local=(ids-tl.load(START)+256)%256
    tl.store(LIDS+t*8+e,tl.where(local<32,local,-1),e<8)
'''
out=src.replace(old,new)+'\nfrom warp_topk import warp_top8\n'
assert ast.dump(next(n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name=='_ps_gemv'))==ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ps_gemv'))
path=HERE/'stats_fast_route.py';path.write_text(out)
print(path,hashlib.sha256(out.encode()).hexdigest())
