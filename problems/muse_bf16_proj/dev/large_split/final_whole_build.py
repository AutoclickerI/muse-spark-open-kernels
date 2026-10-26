"""Judge both rebuilds and native-schedule weight packing by one held-out G0."""
import base64,hashlib,json,lzma
from pathlib import Path
import ss_build,packed_build,ss_source,packed_source
H=Path(__file__).resolve().parent;R=H.parents[3]
r=json.loads((H/'joint_results.json').read_text());p=json.loads((H/'packed_results.json').read_text());assert r['valid'] and p['valid']
sspoints={(t,role):('SS',ss_source.CONFIGS.index(tuple(c)),s) for t,role,c,s in json.loads((H/'ss_points.json').read_text())};pspoints={(t,role):('PS',packed_source.CONFIGS.index(tuple(c)),s) for t,role,c,s in json.loads((H/'packed_points.json').read_text())}
source=packed_build.source
source+='\nimport torch,triton,triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\n_VDSK_TD=globals()["_VDSK_TD"]\nNP_CONFIG='+repr({'qkv_gate':(128,64,64,8,4,1,3,False),'pool_router':(128,128,64,5,8,1,3,True)})+'\n'+(H/'native_pack.py.inc').read_text()
portfolios={'SS':sspoints,'PS':pspoints,'NATIVE_PACK':{(128,r):('NP',0,3) for r in ('qkv_gate','pool_router')}}
source+='\nSS_PORTFOLIOS='+repr(portfolios)+'''\nSS_ACTIVE=False
SS_SELECTED={}
_np_old_prepare=prepare
_np_old_best7_project=globals()["_best7_project"]
_np_old_variant=globals()["_project_variant"]

def prepare(weights):
    st=_np_old_prepare(weights)
    st['ss']=ss_prepare(st)
    st['ps']=ps_prepare(st)
    st['np']=np_prepare(st)
    return st

def _np_dispatch(x,w,name,st):
    family,idx,s=SS_SELECTED[x.shape[0],name]
    if family=='NP':return np_project(x,w,name,st['np'])
    if family=='PS':return ps_project(x,w,name,st['ps'],idx,s)
    return ss_project(x,w,st['ss'],idx,s)

def _best7_project(x,name,st,plan,out_dtype=torch.bfloat16):
    if SS_ACTIVE and 'np' in st and (x.shape[0],name) in SS_SELECTED:return _np_dispatch(x,st[name],name,st)
    return _np_old_best7_project(x,name,st,plan,out_dtype)

def _project_variant(x,w,name,st,variant,out_dtype=torch.bfloat16):
    if SS_ACTIVE and 'np' in st and (x.shape[0],name) in SS_SELECTED:return _np_dispatch(x,w,name,st)
    return _np_old_variant(x,w,name,st,variant,out_dtype)
'''
(H/'generated/final_whole_source.py').write_text(source)
helpers=(R/'problems/muse_decode_layer/dev/s19/graph_helpers.py.inc').read_text();lab=(H/'whole_lab.py.inc').read_text().replace('17000311','19000311').replace('17000313','19000313').replace('17000319','19000319')
lab=lab.replace("make=lambda seed:task['generate_input'](","make=lambda seed:m['_ap_generate_input'](")
lab=lab.replace("if torch.cuda.get_device_capability()!=(9,0):task['_fa3_scheduler_metadata']=lambda *a,**kw:None", "if torch.cuda.get_device_capability()!=(9,0):task['_fa3_scheduler_metadata']=lambda *a,**kw:None;m['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None")
lab=lab.replace("resources=st['ss']['resources']", "resources=dict(ss=st['ss']['resources'],ps=st['ps']['resources'],np=st['np']['resources'])")
lab=lab.replace("added_prepare_s=st['ss']['prepare_s']", "added_prepare_s=dict(ss=st['ss']['prepare_s'],ps=st['ps']['prepare_s'],np=st['np']['prepare_s'],np_pack=st['np']['pack_s'])")
stage="""
 projection={};exec(compile(PROJECTION_CHECK,'<published projection TASK>','exec'),projection);native_projection_checks=0
 for name in ('qkv_gate','pool_router'):
  w=st[name];n,k=w.shape;wp,desc=st['np']['banks'][name];bw=64 if name=='qkv_gate' else 128
  restored=wp.view(k//64,n//bw,bw,64).permute(1,2,0,3).reshape(n,k);assert torch.equal(restored.view(torch.int16),w.view(torch.int16)),('native pack bits',name)
  for seed in [20000319+104729*i for i in range(8)]:
   x=torch.randn((128,k),device='cuda',dtype=torch.bfloat16,generator=torch.Generator(device='cuda').manual_seed(seed))
   for column in (False,True):
    if column:x.zero_();x[:,seed%k]=16
    pair=m['np_project'](x,w,name,st['np']);v=pair[0].sum(0);d=dict(x=x,config=dict(layer='qkv_gate',m=128,n=n,k=k));ok,msg=projection['check_implementation'](d,v.bfloat16(),dict(qkv_gate=w));assert ok,(name,seed,column,msg);native_projection_checks+=1
    if name=='pool_router':
     rd=dict(x=x,config=dict(layer='router',m=128,n=256,k=k));rv=v[:,4096:4352]+v[:,4352:];ok,msg=projection['check_implementation'](rd,rv,weights);assert ok,(name,seed,column,'original FP32 router',msg);native_projection_checks+=1
"""
needle=" def invoke(arm,data):";assert lab.count(needle)==1;lab=lab.replace(needle,stage+'\n'+needle)
lab=lab.replace("weight_seed=19000311", "native_projection_TASK=native_projection_checks,weight_seed=19000311")
# Source uses the unchanged native plan's BT/BW/BK/stages/warps/splits/CHAIN.
# Only the physical weight tile address changes; correctness and full G0 judge it.
payload='LAB_RESOURCES={"compile_s":240,"run_s":280,"output_bytes":8744,"context_poison":"device"}\nimport torch,time,types,linecache,statistics,math,json,hashlib,base64,lzma\nWHOLE_SOURCE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(source.encode())).decode())+')).decode()\nWHOLE_TASK='+repr(ss_build.tasks)+'\nPROJECTION_CHECK='+repr(ss_build.checker)+'\n'+helpers+'\n'+lab
compile(payload,'<P10 final heldout whole>','exec');(H/'generated/final_whole_payload.py').write_text(payload)
(H/'final_whole_manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(ss_build.base.encode()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),arms=list(portfolios),seeds=[19000319+104729*i for i in range(16)],fixed_native_points=['T128 qkv_gate','T128 pool_router'],selection_receipts=[r['job'],p['job']]),indent=2)+'\n');print(len(payload))
