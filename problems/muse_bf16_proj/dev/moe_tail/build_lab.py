"""Pack all shell arms in one S15-host prepare-phase paired whole-layer lab."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def functions(path,names):
 s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)
def main():
 trace=(R/'problems/muse_decode_layer/dev/s15_timeline/timeline_generated.py.inc').read_text();activity=next(ast.literal_eval(n.value) for n in ast.parse(trace).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='T15_ACTIVITY')
 body='import ast,base64,gc,hashlib,json,lzma,math,statistics,struct,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
 body+='MT_RUNTIME='+repr((H/'runtime_generated.py.inc').read_text())+'\nT15_ACTIVITY='+repr(activity)+'\nT15_REPS=3\n'
 body+=(R/'problems/muse_decode_layer/dev/s15_timeline/analysis.py').read_text()+'\n'
 body+=functions(R/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc',('ab_copy','ab_tensors','ab_gate'))+'\n'
 body+=functions(R/'problems/muse_decode_layer/dev/s15_timeline/lab_body.inc',('t15_activity','t15_namespace','t15_capture','t15_profile','t15_signature','t15_busiest'))+'\n'
 body+=(H/'lab_body.inc').read_text();compile(body,'moe_tail_lab','exec');tag=hashlib.sha256(body.encode()).hexdigest();encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
 boot='import base64,lzma,linecache,hashlib\n_MT_BODY=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_MT_BODY.encode()).hexdigest()=='+repr(tag)+'\n_MT_FILE="<p7-moe-tail-'+tag[:12]+'>"\nlinecache.cache[_MT_FILE]=(len(_MT_BODY),None,_MT_BODY.splitlines(True),_MT_FILE)\nexec(compile(_MT_BODY,_MT_FILE,"exec"),globals())\n_MT_RUN=lab\ndef lab(deadline,weights,state=None):\n    return _MT_RUN(deadline,weights,state)\n'
 (H/'lab_generated.py.inc').write_text(body);(H/'lab_moe_tail.py').write_text(boot);(H/'lab_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(boot.encode()).hexdigest(),body_sha256=tag),indent=2)+'\n');print('payload',hashlib.sha256(boot.encode()).hexdigest()[:8])
if __name__=='__main__':main()
