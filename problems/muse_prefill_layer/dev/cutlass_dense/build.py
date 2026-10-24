"""Freeze the proj CUTLASS core into a prefill lab and a new single-file portfolio."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'problems/muse_prefill_layer/cand_mid_dense.py'
base=BASE.read_text()

def definition(name):
    node=next(n for n in ast.parse(base).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    start=node.decorator_list[0].lineno if isinstance(node,ast.FunctionDef) and node.decorator_list else node.lineno
    return '\n'.join(base.splitlines()[start-1:node.end_lineno])+'\n'

source=(HERE/'kernel.cu').read_text()
core=(HERE/'runtime.py.inc').read_text()+'\nVGC_SOURCE = '+repr(source)+'\n'
common=(HERE/'common.py.inc').read_text()
candidate=base.replace('    w=st[key];fp32=key==\'router_cat\';name=choice[0]',
    "    if choice[0]=='cutlass':return pfc_run(x,st,key,choice[1])\n    w=st[key];fp32=key=='router_cat';name=choice[0]",1)
candidate=candidate.replace('        _tune_score_attention(st,flush)',
    "        _tune_score_attention(st,flush)\n        pfc_prepare(st,min(st['_deadline'],started+245))",1)
candidate=candidate.replace("    print('SCORE_LAST_SUMMARY',", "    for row in st.get('_pfc_notes',[]):print(row,flush=True)\n    print('SCORE_LAST_SUMMARY',",1)
candidate += '\n'+core+'\n'+common
out=ROOT/'problems/muse_prefill_layer/cand_mid_cutlass.py'
out.write_text(candidate)

# Isolate the exact incumbent's Lt32 heuristic ABI, without compiling experts.
tree=ast.parse(base)
grouped=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and
    any(isinstance(t,ast.Name) and t.id=='GROUPED_SRC' for t in n.targets))
lt=grouped[grouped.index('#include <cublasLt.h>'):]
lt='#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n'+lt
lt+='\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("lt_options",&lt_options);m.def("lt_run",&lt_run);}\n'
receipt=json.loads((ROOT/'runs/muse_prefill_layer/20261024T043516Z_submit_0d738799802340fd.json').read_text())
line=next(l for l in receipt['result']['log_tail'].splitlines() if l.startswith('SCORE_PREPARE '))
choices=json.loads(line.split(' ',1)[1])['dense']
choices={ast.literal_eval(k):tuple(v) for k,v in choices.items()}
imports='import ctypes as ct\nimport torch,time,statistics,json,hashlib,os\nimport triton\nimport triton.language as tl\n'
triton_config="_DENSE_TRITON={'triton64':(64,128,64,4,3),'triton128':(128,128,64,8,3),'triton256':(128,256,64,8,3)}\n"
payload=imports+core+'\n'+definition('_PgGraph')+'\n'+common+'\n'+definition('_dense_persistent')+'\n'+triton_config+definition('_score_dense_run')
payload+='\nPFC_LT_SOURCE = '+repr(lt)+'\nPFC_INCUMBENTS = '+repr(choices)+'\n'+(HERE/'lab.py.inc').read_text()
lab=ROOT/'prof/lab/prefill_dense_cutlass.py'
lab.write_text(payload)
manifest={'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
          'cutlass_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'candidate':str(out.relative_to(ROOT)),'candidate_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'payload':str(lab.relative_to(ROOT)),'payload_sha256':hashlib.sha256(lab.read_bytes()).hexdigest()}
(HERE/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
