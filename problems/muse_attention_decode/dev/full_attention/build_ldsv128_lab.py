"""Freeze both native N128 schedules and their measured two-head N128 comparator."""
import ast
import subprocess
import sys
from pathlib import Path
H=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(H/'build_ldsv128.py')],check=True)
subprocess.run([sys.executable,str(H/'build_paired.py'),'--parent','ldsv128','--name','ldsv128pair'],check=True)
p=H/'ldsv128_lab.py';s=p.read_text();tree=ast.parse(s)
n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SOURCES')
src=ast.literal_eval(n.value);src['pair128']=(H/'ldsv128pair.py').read_text()
lines=s.splitlines(keepends=True);lines[n.lineno-1:n.end_lineno]=['SOURCES='+repr(src)+'\n'];s=''.join(lines)
s=s.replace('range(7)','range(10)').replace('range(5)','range(8)')
old="+[lambda:mods['pair64']['vq8_attention'](d,stages=4,min_tiles=1)]"
assert s.count(old)==1
s=s.replace(old,"+[lambda st=st:mods['pair128']['vq8_attention'](d,stages=st,min_tiles=1) for st in (2,3,4)]"+old)
s=s.replace("('best128','new2','new3','new4','pair64')","('best128','new2','new3','new4','pair2','pair3','pair4','pair64')")
p.write_text(s)
