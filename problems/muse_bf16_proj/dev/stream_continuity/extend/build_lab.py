import ast,base64,hashlib,importlib.util,json,lzma
from pathlib import Path
from build import build,ARMS
H=Path(__file__).resolve().parent;R=H.parents[4]
def funcs(path,names):
 s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)
def main():
 sources=build();old=H.parent;spec=importlib.util.spec_from_file_location('old_builder',old/'build_lab.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 timeline=R/'problems/muse_decode_layer/dev/s15_timeline';gen=(timeline/'timeline_generated.py.inc').read_text();activity=next(ast.literal_eval(n.value) for n in ast.parse(gen).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='T15_ACTIVITY')
 bundle=base64.b85encode(lzma.compress(json.dumps(sources,separators=(',',':')).encode(),preset=9)).decode()
 body='import ast,base64,gc,hashlib,json,linecache,lzma,math,statistics,struct,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
 body+='SC_SOURCES=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nSC_CHECKER='+repr(mod.checker())+'\nEW_CONFIG='+repr(ARMS)+'\nT15_ACTIVITY='+repr(activity)+'\nT15_REPS=2\n'
 body+=(timeline/'analysis.py').read_text()+'\n_ew_old_stage=stage_name\ndef stage_name(role,name):\n    return "moe_heavy" if "chunk::kernel" in name else _ew_old_stage(role,name)\n'
 body+=funcs(R/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc',('ab_copy','ab_tensors'))+'\n'
 body+=funcs(timeline/'lab_body.inc',('t15_activity','t15_capture','t15_profile','t15_signature','t15_busiest'))+'\n'
 body+=funcs(old.parent/'residual_tail/lab_body.inc',('rt_task_fixture',))+'\n'+funcs(old/'setup.py.inc',('sc_namespace',))+'\n'
 body+=(H/'setup.py.inc').read_text()+'\n'+(H/'lab_body.inc').read_text();compile(body,'ew_lab','exec');sha=hashlib.sha256(body.encode()).hexdigest()
 encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode();boot='LAB_RESOURCES={"compile_s":360,"run_s":480,"output_bytes":8744,"context_poison":"none"}\nimport base64,hashlib,linecache,lzma\n_EW_BODY=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_EW_BODY.encode()).hexdigest()=='+repr(sha)+'\n_EW_FILE="<p7-owned-'+sha[:12]+'>"\nlinecache.cache[_EW_FILE]=(len(_EW_BODY),None,_EW_BODY.splitlines(True),_EW_FILE)\nexec(compile(_EW_BODY,_EW_FILE,"exec"),globals())\n_EW_RUN=globals()["lab"]\ndef lab(deadline,weights,state=None):return _EW_RUN(deadline,weights,state)\n'
 (H/'lab_generated.py.inc').write_text(body);(H/'lab_owned.py').write_text(boot);(H/'lab_manifest.json').write_text(json.dumps(dict(body_sha256=sha,payload_sha256=hashlib.sha256(boot.encode()).hexdigest(),arm_sha256={a:hashlib.sha256(s.encode()).hexdigest() for a,s in sources.items()},seeds=[37841051+15485863*i for i in range(16)],TASK_checks=3456,host='muse_prefill_layer',phase='import',budget_s=840),indent=2)+'\n');print('payload',sha)
if __name__=='__main__':main()
