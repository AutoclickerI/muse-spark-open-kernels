"""Adjacent N32 token groups share a weight tile in L2; tile body unchanged.

Apply onto S12 final, or compose after PERMW: python3 SCRIPT INPUT OUTPUT.
Only task indexing changes from (expert, token-group, channel-tile) to
(expert, channel-tile, token-group). No additional padding or shared storage.
"""
import ast
import hashlib
import sys
from pathlib import Path


def patch(text):
    n=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NF_CUDA')
    cu=ast.literal_eval(n.value)
    old='  int e=ORDER[ix],row=(group-TASK[ix])*32,count=C[e],offset=OFF[e];'
    new='''  int e=ORDER[ix],nb=TASK[ix+1]-TASK[ix],rel=unit-TASK[ix]*MT;
  mt=rel/nb;int row=(rel-mt*nb)*32,count=C[e],offset=OFF[e];'''
    assert cu.count(old)==1,'expected S12 N32 task map'
    cu=cu.replace(old,new)
    lines=text.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['NF_CUDA='+repr(cu)+'\n']
    text=''.join(lines);compile(text,'<S12 weight reuse order>','exec');return text


if __name__=='__main__':
    src,dst=map(Path,sys.argv[1:]);dst.write_text(patch(src.read_text()))
    print(dst,hashlib.sha256(dst.read_bytes()).hexdigest())
