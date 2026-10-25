"""Freeze one-warp decode, cluster-size choices, and preparation/gather/DSM clocks into S11's comparator."""
import ast,hashlib,lzma,base64
from pathlib import Path
import build_warp,build_clock
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(ROOT/'prof/lab/cluster_sliding_timeline.py').read_text()
lines=s.splitlines(True);tree=ast.parse(s)
for n in reversed(tree.body):
    if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('cs_CUDA','TL_S11'):
        key=n.targets[0].id
        val=repr(build_warp.source()) if key=='cs_CUDA' else 'lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(ast.literal_eval(n.value).encode())).decode())+')).decode()'
        lines[n.lineno-1:n.end_lineno]=[key+'='+val+'\n']
s=''.join(lines)
s=s.replace('CS_CASES=tuple((b,q) for q in (1,8) for b in (1,16,32))','CS_CASES=tuple((b,1) for b in (1,16,32))')
s=s.replace('def cs_namespace(host,nk):','def cs_namespace(host,nk,cs=None):').replace('qkv,nk=nk).view(-1,host.Q_SIZE)','qkv,nk=nk,cs=cs).view(-1,host.Q_SIZE)',1)
s=s.replace("'C128':cs_namespace(host,128)}", "'C128':cs_namespace(host,128),'W4':cs_namespace(host,128,4),'W8':cs_namespace(host,128,8)}")
s=s.replace("for n in ('C64','C128')", "for n in ('C64','C128','W4','W8')")
s=s.replace('C64/C128=cluster DSM reduction with native FP8 QK/RS PV','C64/C128=32-thread m16 FP8 MMA + DSM, auto split; W4/W8 force 4/8 splits')
s=s.replace('END shapes=%d/6','END shapes=%d/3')
s=s.replace("            del qkv,got", "            phase_rows=[cs_phase(qkv,data,w,host,st) for w in (False,True)]\n            lines.append('PH '+label+' '+json.dumps(phase_rows,separators=(',',':')))\n            del qkv,got")
s=s.replace('max_workers=2','max_workers=4').replace('af=pool.submit(tl_activity);kf=pool.submit(cs_module)', 'af=pool.submit(tl_activity);kf=pool.submit(cs_module);clock_fs=[pool.submit(cs_clock_module,w) for w in (False,True)]').replace('act=af.result();kf.result()', 'act=af.result();kf.result()\n        for f in clock_fs:f.result()')
a=s.index('def lab(deadline,weights,state=None):')
s=s[:a]+'CS_CLOCK_CUDA='+repr({w:build_clock.source(w) for w in (False,True)})+'\n'+(HERE/'clock_runtime.py.inc').read_text()+'\n'+s[a:]
compile(s,'warp_sliding_timeline.py','exec');p=ROOT/'prof/lab/warp_sliding_timeline.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s))
