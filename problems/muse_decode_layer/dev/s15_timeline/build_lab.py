"""Instrument the host's exact S15 function globals without changing any device kernel."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
S15_SHA='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
def functions(path,names):
 s=path.read_text();return '\n\n'.join(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in names)
def main():
 source=ROOT/'problems/muse_decode_layer/cand_r3_best7_s15.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==S15_SHA
 activity=(HERE.parent/'timeline/activity.cpp').read_text()
 marker='   }\n }\n size_t lost=0;'
 extra='   } else if(record->kind==CUPTI_ACTIVITY_KIND_MEMSET){\n     auto* k=(CUpti_ActivityMemset4*)record;\n     activity_rows.emplace_back("cudaMemset",std::vector<int64_t>{(int64_t)k->start,(int64_t)k->end,0,0,(int64_t)k->graphNodeId,0,(int64_t)k->graphId,(int64_t)k->bytes,1,1,0,0,0,k->streamId});\n   } else if(record->kind==CUPTI_ACTIVITY_KIND_MEMCPY){\n     auto* k=(CUpti_ActivityMemcpy6*)record;\n     activity_rows.emplace_back("cudaMemcpy",std::vector<int64_t>{(int64_t)k->start,(int64_t)k->end,0,0,(int64_t)k->graphNodeId,0,(int64_t)k->graphId,(int64_t)k->bytes,1,1,0,0,0,k->streamId});\n   }\n }\n size_t lost=0;'
 assert activity.count(marker)==1;activity=activity.replace(marker,extra)
 activity=activity.replace('if(type==cudaGraphNodeTypeKernel){','if(type==cudaGraphNodeTypeKernel||type==cudaGraphNodeTypeMemset||type==cudaGraphNodeTypeMemcpy){')
 activity=activity.replace('ck(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));','ck(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));ck(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));ck(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));')
 activity=activity.replace('activity_on=false;\n auto latency_status=', 'activity_on=false;\n ck(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));ck(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));\n auto latency_status=')
 dag=(ROOT/'problems/muse_bf16_proj/dev/boundary_attribution/dag.cpp.inc').read_text()
 dag=dag.replace('type==cudaGraphNodeTypeKernel','(type==cudaGraphNodeTypeKernel||type==cudaGraphNodeTypeMemset||type==cudaGraphNodeTypeMemcpy)')
 activity+='\n'+dag
 helpers=functions(HERE.parent/'s10/ablate_lab.py.inc',('ab_copy','ab_graph','ab_tensors','ab_gate','ab_time'))
 body='import base64,gc,hashlib,json,lzma,math,statistics,struct,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
 body+='T15_SOURCE_SHA='+repr(S15_SHA)+'\nT15_ACTIVITY='+repr(activity)+'\n'+(HERE/'analysis.py').read_text()+'\n'+helpers+'\n'
 body+=(HERE/'footprints.py.inc').read_text()+'\n'+(HERE/'lab_body.inc').read_text();compile(body,'s15_timeline_lab','exec');tag=hashlib.sha256(body.encode()).hexdigest()
 (HERE/'timeline_generated.py.inc').write_text(body);encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
 boot='import base64,lzma,linecache,hashlib\n_T15_BODY=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_T15_BODY.encode()).hexdigest()=='+repr(tag)+'\n_T15_FILE="<s15-timeline-'+tag[:12]+'>"\nlinecache.cache[_T15_FILE]=(len(_T15_BODY),None,_T15_BODY.splitlines(True),_T15_FILE)\nexec(compile(_T15_BODY,_T15_FILE,"exec"),globals())\n_T15_RUN=lab\ndef lab(deadline,weights,state=None):\n    return _T15_RUN(deadline,weights,state)\n'
 (HERE/'lab_s15_timeline.py').write_text(boot);(HERE/'manifest.json').write_text(json.dumps(dict(s15_sha256=S15_SHA,activity_sha256=hashlib.sha256(activity.encode()).hexdigest(),payload_sha256=hashlib.sha256(boot.encode()).hexdigest(),body_sha256=tag),indent=2)+'\n');print('payload',hashlib.sha256(boot.encode()).hexdigest())
if __name__=='__main__':main()
