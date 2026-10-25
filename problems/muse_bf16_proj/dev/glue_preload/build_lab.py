import ast
import base64
import hashlib
import lzma
from pathlib import Path
import sys

from patch import BASE_SHA,cuda_source

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO/'tools'))
import h200_lab


def pieces(src,names):
    return '\n\n'.join(ast.get_source_segment(src,n) for n in ast.parse(src).body
        if isinstance(n,ast.FunctionDef) and n.name in names)


base=(REPO/'problems/muse_decode_layer/cand_r3_best7_s13_final.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()==BASE_SHA
runtime=(HERE/'runtime.py.inc').read_text().replace('__GP_CUDA__',repr(cuda_source(base)),1).replace('__GP_ROWS__','()',1)
root=pieces(base,('_ap_generate_input','_best7_run','kernel_fn'))
old="    f=_best7_project(moe,name,state,plan)\n    hidden,stream2=_best7_boundary(f,stream1,state,plan,'b2',sync[2])"
assert root.count(old)==1
root=root.replace(old,"    if state.get('epb_op') is not None:\n        hidden,stream2=_gp_epb(moe,stream1,state,plan)\n    else:\n    "+old.replace('\n','\n    '))
ep=HERE.parent/'producer_glue'
epcode=(ep/'kernels.py').read_text()+'\n'+(ep/'stats_parallel.py').read_text().replace('from kernels import ProducerGlue, _pg_wait, _pg_clock, _pg_meta\n','')
ab=(REPO/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc').read_text()
tl=REPO/'problems/muse_decode_layer/dev/timeline'
body='import base64,json,lzma,types,linecache,time,statistics,gc\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
body+='GP_RUNTIME='+repr(runtime)+'\nGP_ROOT='+repr(root)+'\nEP_CODE='+repr(epcode)+'\nTL_ACTIVITY='+repr((tl/'activity.cpp').read_text())+'\nTL_REPS=5\n'
body+=pieces(ab,('ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','ab_summary'))+'\n'
body+=pieces((tl/'lab_body.py').read_text(),('tl_activity','tl_times','tl_union','tl_summarize','tl_node_bytes'))+'\n'
body+=(HERE/'lab_body.py.inc').read_text();compile(body,'gp_lab','exec')
(HERE/'lab_body_generated.py').write_text(body)
tag=hashlib.sha256(body.encode()).hexdigest()
bootstrap='import base64,lzma,linecache,hashlib\n_GP_BODY=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(body.encode(),preset=9)).decode())+')).decode()\n'
bootstrap+='assert hashlib.sha256(_GP_BODY.encode()).hexdigest()=='+repr(tag)+'\n_GP_FILE="<gp-lab-'+tag[:12]+'>"\n'
bootstrap+='linecache.cache[_GP_FILE]=(len(_GP_BODY),None,_GP_BODY.splitlines(True),_GP_FILE)\nexec(compile(_GP_BODY,_GP_FILE,"exec"),globals())\n_GP_LAB=lab\ndef lab(deadline,weights,state):\n    return _GP_LAB(deadline,weights,state)\n'
payload=HERE/'lab_preload.py';payload.write_text(bootstrap)
wrapped=base.rstrip()+'\n\n'+h200_lab.WRAPPER.format(names='P7 preload repair',base='S13 958f1258',
    payloads=[h200_lab.embed(payload)],budget=300-h200_lab.MARGIN_S,limit=h200_lab.LIMIT)
out=HERE/'preload_lab_trial.py';out.write_text(wrapped)
print('payload',hashlib.sha256(bootstrap.encode()).hexdigest(),'body',tag)
print(out,len(wrapped),hashlib.sha256(wrapped.encode()).hexdigest())
