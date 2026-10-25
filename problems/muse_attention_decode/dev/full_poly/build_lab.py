"""Freeze the two polynomial variants into the original S11 critical-path comparator."""
import ast,hashlib,lzma,base64
from pathlib import Path
import build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];TL=ROOT/'problems/muse_decode_layer/dev/timeline'
s11=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text();assert hashlib.sha256(s11.encode()).hexdigest().startswith('acda6c62')
node=next(n for n in ast.parse(s11).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CPP')
cpp=ast.literal_eval(node.value)
drv=(HERE.parent/'vq8/vq8.py').read_text();summary=drv[drv.index('def clock_summary('):drv.index('def attention_vq8(')]
shared=(HERE.parent/'cluster_sliding/lab.py.inc').read_text();shared=shared[shared.index('def cs_copy('):shared.index('def lab(deadline,')]
src=('import gc, hashlib, json, linecache, math, statistics, time, types, lzma, base64\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
 +'TL_S11=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(s11.encode())).decode())+')).decode()'+'\nTL_ACTIVITY='+repr((TL/'activity.cpp').read_text())+'\nPF_CUDA='+repr({d:build.source(d) for d in (2,3)})+'\nPF_CPP='+repr(cpp)+'\n'
 +(TL/'selected_prepare.py').read_text()+'\n'+(TL/'lab_body.py').read_text().split('def lab(deadline, weights, state=None):')[0]+'\n'+shared+'\n'+summary+'\n'+(HERE/'lab.py.inc').read_text())
compile(src,'full_poly_timeline.py','exec');p=ROOT/'prof/lab/full_poly_timeline.py';p.write_text(src);print(p,hashlib.sha256(src.encode()).hexdigest())
