"""Reuse the verified S11 comparator for N128 stage3/stage4 vs frozen S11 N64 stage6."""
import hashlib,lzma,base64,ast
from pathlib import Path
import build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];TL=ROOT/'problems/muse_decode_layer/dev/timeline'
s11=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text();assert hashlib.sha256(s11.encode()).hexdigest().startswith('acda6c62')
n=next(n for n in ast.parse(s11).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CPP');cpp=ast.literal_eval(n.value)
drv=(HERE.parent/'vq8/vq8.py').read_text();summary=drv[drv.index('def clock_summary('):drv.index('def attention_vq8(')]
shared=(HERE.parent/'cluster_sliding/lab.py.inc').read_text();shared=shared[shared.index('def cs_copy('):shared.index('def lab(deadline,')]
body=(HERE.parent/'full_poly/lab.py.inc').read_text()
body=body.replace('key=d,clk','key=clk').replace('PF_CUDA[d]','PF_CUDA').replace("f'vq8_probpoly{d}_{tag}_{arch}_{int(clk)}'","f'vq8_full128_{tag}_{arch}_{int(clk)}'")
body=body.replace('for d in (2,3)','for d in (3,4)').replace('st=6 if','st=d if')
body=body.replace("'P2':pf_namespace(host,2),'P3':pf_namespace(host,3)","'T3':pf_namespace(host,3),'T4':pf_namespace(host,4)")
body=body.replace('FP32 polynomial probability exponent vs native ex2; rescale/max stays native FP32 ex2.','N128 FP8 QK / register RS PV, stages3/4 vs incumbent N64 stage6; native FP32 ex2 throughout.')
body=body.replace('PF_RESULTS=[]\nPF_CLOCKS=True','PF_RESULTS=[]\nPF_CLOCKS=True')
src=('import gc, hashlib, json, linecache, math, statistics, time, types, lzma, base64\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
 +'TL_S11=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(s11.encode())).decode())+')).decode()\nTL_ACTIVITY='+repr((TL/'activity.cpp').read_text())+'\nPF_CUDA='+repr(build.source())+'\nPF_CPP='+repr(cpp)+'\n'
 +(TL/'selected_prepare.py').read_text()+'\n'+(TL/'lab_body.py').read_text().split('def lab(deadline, weights, state=None):')[0]+'\n'+shared+'\n'+summary+'\n'+body)
compile(src,'full128_timeline.py','exec');p=ROOT/'prof/lab/full128_timeline.py';p.write_text(src);print(p,hashlib.sha256(src.encode()).hexdigest(),len(src))
