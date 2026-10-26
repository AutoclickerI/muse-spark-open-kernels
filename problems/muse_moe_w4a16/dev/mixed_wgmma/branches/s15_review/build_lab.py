"""Pack independent S15 arms; retain seed means/SE and bounded A:A transport."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib
spec=importlib.util.spec_from_file_location('p3b_s15_builder',Path(__file__).with_name('build.py'))
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
H,B,BASE,source,function,value=(getattr(builder,k) for k in ('H','B','BASE','source','function','value'))

def main():
 builder.prepare_parent()
 r6=(B/'round6/p3b_repair_payload.py').read_text();base=BASE.read_text()
 # Only synthetic fixtures, exact bank preparation, and control plan helpers survive here.
 body=r6[:r6.index('SAFE_CUDA=')]
 body+='\nNF_CUDA='+repr(value(base,'NF_CUDA'))+'\nNF_CPP='+repr(value(base,'NF_CPP'))+'\nNF_OPTS={}\n'
 body+=function(base,'nf_module')+'\n'+function(base,'nf_plan')+'\n'+function(base,'nf_call')+'\n'
 body+=function(r6,'op_error')+'\n'+function(r6,'op_pair')+'\n'
 cells={arm:source(arm)[1].split('_p3b_original_s4_prepare_banks=_s4_prepare_banks')[0] for arm in ('CHUNK3','CHUNK4','PACK5','READY_SERIAL','READY')}
 body+='\nREVIEW_CELLS='+repr(cells)+'\n'+(H/'lab.py.inc').read_text()
 compile(body,'<P3b S15 review>','exec')
 # Source packing changes no executable source and keeps the host wrapper below1MB.
 packed='import base64,zlib\nexec(compile(zlib.decompress(base64.b85decode('+repr(base64.b85encode(zlib.compress(body.encode(),9)).decode())+')),"<p3b-s15-review>","exec"))\n'
 packed+='\ndef lab(deadline,weights,state=None):return review_lab(deadline,weights,state)\n'
 (H/'s15_review_payload.py').write_text(packed)
 print(dict(payload_sha256=hashlib.sha256(packed.encode()).hexdigest(),expanded_sha256=hashlib.sha256(body.encode()).hexdigest(),bytes=len(packed)))
 (H/'payload.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(packed.encode()).hexdigest(),expanded_sha256=hashlib.sha256(body.encode()).hexdigest(),bytes=len(packed),arms=['S15',*cells]),indent=2)+'\n')

if __name__=='__main__':main()
