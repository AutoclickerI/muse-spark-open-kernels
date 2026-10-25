"""Pack decode-1 glue attribution into one self-preparing synthetic H200 lab."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
from patch import patch,previous,MODES
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def functions(path,names):
 s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)
def main():
 base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
 sources={n:patch(base,n) for n in MODES};sources['D']=previous.patch(base)
 for n,s in sources.items():(HERE/(n+'_trial.py')).write_text(s)
 synth='HIDDEN=8192;Q_SIZE=1024;QKV_SIZE=2304;D_EXPERT=4096;LOCAL_EXPERTS=32;N_EXPERTS=256\nROUTER_STD=.045;ROUTER_ALIGN=.07;ROUTER_BIAS_STD=.055;STREAM_SEED=5354;STREAM_RANK=4\nBLOCK=16;E4M3_MAX=448.;SCALE_LOG2_MEAN=7.;SCALE_LOG2_STD=.4\n'
 synth+=functions(ROOT/'problems/muse_decode_layer/nvfp4.py',('random_weights',))+'\n'
 synth+=functions(ROOT/'problems/muse_decode_layer/reference.py',('_stream_basis','_router','generate_weights'))+'\n'
 helpers=functions(ROOT/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc',('ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','ab_summary'))
 helpers+='\n'+functions(HERE.parent/'pdl_audit/lab_body.inc',('pa_namespace','pa_assignments','pa_state'))
 helpers+='\n'+functions(HERE.parent/'pdl_audit/followup_body.inc',('pa_share','pa_share_jit','pa_quality'))
 helpers+='\n'+functions(HERE.parent/'boundary_attribution/lab_body.inc',('bd_capture',))
 activity=(ROOT/'problems/muse_decode_layer/dev/timeline/activity.cpp').read_text()+'\n'+(HERE.parent/'boundary_attribution/dag.cpp.inc').read_text()
 body='import ast,base64,functools,gc,hashlib,json,linecache,lzma,math,statistics,struct,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
 body+='PA_BASE='+repr(base)+'\nDC_SOURCES='+repr(sources)+'\nDC_ACTIVITY='+repr(activity)+'\n'+synth+helpers+'\n'
 body+=(HERE/'lab_body.inc').read_text();compile(body,'d1_chain_lab','exec')
 (HERE/'chain_generated.py.inc').write_text(body);tag=hashlib.sha256(body.encode()).hexdigest();encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
 boot='import base64,lzma,linecache,hashlib\n_DC_BODY=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_DC_BODY.encode()).hexdigest()=='+repr(tag)+'\n_DC_FILE="<d1-chain-'+tag[:12]+'>"\nlinecache.cache[_DC_FILE]=(len(_DC_BODY),None,_DC_BODY.splitlines(True),_DC_FILE)\nexec(compile(_DC_BODY,_DC_FILE,"exec"),globals())\n_DC_RUN=lab\ndef lab(deadline,weights,state=None):\n    return _DC_RUN(deadline,weights,state)\n'
 (HERE/'lab_d1_chain.py').write_text(boot)
 manifest=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),payload_sha256=hashlib.sha256(boot.encode()).hexdigest(),body_sha256=tag,variants={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()})
 (HERE/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
