"""Single S15-host whole-layer paired batch, exact copied public TASK checker per arm/seed."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def functions(path,names):
 s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)
def checker():
 sources=[(R/'problems/muse_decode_layer'/name).read_text() for name in ('nvfp4.py','checks.py','reference.py')]
 funcs={n.name:(s,n) for s in sources for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)};needed={'collect','check_implementation','ref_front','ref_back'}
 while True:
  more={n.id for name in needed for n in ast.walk(funcs[name][1]) if isinstance(n,ast.Name) and n.id in funcs}
  if more<=needed:break
  needed|=more
 result='import math,functools\nimport torch\n'
 for s in sources:
  for n in ast.parse(s).body:
   if isinstance(n,(ast.Assign,ast.AnnAssign)):result+=ast.get_source_segment(s,n)+'\n'
 for name in funcs:
  if name in needed:
   s,n=funcs[name];result+=ast.get_source_segment(s,n)+'\n'
 compile(result,'public-task-checker','exec');return result

def main():
 t15=(R/'problems/muse_decode_layer/dev/s15_timeline/timeline_generated.py.inc').read_text();activity=next(ast.literal_eval(n.value) for n in ast.parse(t15).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='T15_ACTIVITY')
 body='import ast,base64,gc,hashlib,json,lzma,math,statistics,struct,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
 body+='RT_RUNTIME='+repr((H/'runtime_generated.py.inc').read_text())+'\nRT_TASK_SOURCE='+repr(checker())+'\nT15_ACTIVITY='+repr(activity)+'\nT15_REPS=3\n'
 body+=(R/'problems/muse_decode_layer/dev/s15_timeline/analysis.py').read_text()+'\n'
 body+=functions(R/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc',('ab_copy','ab_tensors'))+'\n'
 body+=functions(R/'problems/muse_decode_layer/dev/s15_timeline/lab_body.inc',('t15_activity','t15_namespace','t15_capture','t15_profile','t15_signature','t15_busiest'))+'\n'
 body+=functions(H.parent/'moe_tail/lab_body.inc',('mt_pair',)).replace('MT_REPS','RT_REPS')+'\n'
 body+=(H/'lab_body.inc').read_text();compile(body,'residual_tail_lab','exec');tag=hashlib.sha256(body.encode()).hexdigest();encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
 boot='import base64,lzma,linecache,hashlib\n_RT_BODY=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_RT_BODY.encode()).hexdigest()=='+repr(tag)+'\n_RT_FILE="<p7-residual-tail-'+tag[:12]+'>"\nlinecache.cache[_RT_FILE]=(len(_RT_BODY),None,_RT_BODY.splitlines(True),_RT_FILE)\nexec(compile(_RT_BODY,_RT_FILE,"exec"),globals())\n_RT_RUN=lab\ndef lab(deadline,weights,state=None):\n    return _RT_RUN(deadline,weights,state)\n'
 (H/'lab_generated.py.inc').write_text(body);(H/'task_generated.py.inc').write_text(checker());(H/'lab_residual_tail.py').write_text(boot);(H/'lab_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(boot.encode()).hexdigest(),body_sha256=tag,task_checker_sha256=hashlib.sha256(checker().encode()).hexdigest()),indent=2)+'\n');print('payload',hashlib.sha256(boot.encode()).hexdigest()[:8])
if __name__=='__main__':main()
