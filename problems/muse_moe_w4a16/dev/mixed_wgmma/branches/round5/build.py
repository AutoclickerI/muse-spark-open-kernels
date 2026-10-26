"""Shared raw vs shared decode at fixed two-N32 token-group ownership."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;B=H.parent;sys.path.insert(0,str(B))
spec=importlib.util.spec_from_file_location('p3b_round4_builder',B/'round4/build.py');r4=importlib.util.module_from_spec(spec);spec.loader.exec_module(r4)
from light_build import value
def fun(src,name):return ast.get_source_segment(src,[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name][-1])
def main():
 # Original frozen queue snapshots are independent of regenerated scratch.
 r4.main()
 src=(B/'round4/ws_payload.py').read_text();ctrl=(B/'round2/cta2.cu').read_text();cpp=value(src,'WS_CPP').split('void ws_gather')[0]
 cu=ctrl+'\n'+(H/'shared.cu.inc').read_text();cpp+='void tp_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> tp_info(int64_t,int64_t);'
 plan=fun(src,'nf_light_plan').replace('def nf_light_plan(','def tp_plan(')
 plan=plan.replace('mod.br_run(p13, 26, grid, m, side.cuda_stream)','mod.tp_run(p13, 1, OP_ARM==3, grid, m, side.cuda_stream)').replace('mod.br_run(p2, 21, grid, m, side.cuda_stream)','mod.tp_run(p2, 0, OP_ARM==3, grid, m, side.cuda_stream)')
 plan=plan.replace('grid = 1056 if m <= 128 else 2112','grid = sms * 4')
 assert 'tp_run' in plan
 prefix=src[:src.index('WS_CUDA=')]
 body=prefix+'\nTP_CUDA='+repr(cu)+'\nTP_CPP='+repr(cpp)+'\nNF_OPTS={}\n'+fun(src,'reuse_plan')+'\n'+plan+'\n'
 body+='\n'.join(fun(src,n) for n in ('nf_module','op_call','op_error','op_pair'))+'\n'
 rt=(B/'round4/runtime.py.inc').read_text();rt=rt[rt.index('def lab('):]
 rt=rt.replace("lines=['META S14 / CTA2", "lines=['META S14 / CTA2")
 start=rt.index(" lines=['META ");end=rt.index('\n for m in (128,256):',start)
 rt=rt[:start]+" lines=['META S14 / CTA2 vs RS_RAW_SHARE and SS_DECODE_SHARE: identical two-N32 token-group ownership, once-per-CTA packed raw copy; SS decodes once into swizzled BF16 A shared by both WGs. No activation pack, cluster, separate producer or global queue. TASK/cold-read direct ABBA pairs against BOTH controls.']"+rt[end:]
 rt=rt.replace("lines.append('WS_RES '+str([(n,f,p,list(OP_MODS[0].ws_info(n,f,p))) for n in (64,128,192,256) for f in (0,1) for p in (0,1)]))","lines.append('TP_RES '+str([(f,s,list(OP_MODS[0].tp_info(f,s))) for f in (0,1) for s in (0,1)]))")
 body+='''OP_MODS=[]
OP_ARM=0
OP_NAMES=('S14','CTA2','RS_RAW_SHARE','SS_DECODE_SHARE')
def op_build():
 if OP_MODS:return
 fs=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run','oc_run','oc_info','oc_gather','tp_run','tp_info']
 if torch.cuda.get_device_capability()==(9,0):
  mod=load_inline(name='p3b_tp_'+hashlib.sha256(TP_CUDA.encode()).hexdigest()[:12],cpp_sources=TP_CPP,cuda_sources=TP_CUDA,functions=fs,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'])
 else:
  mod=load_inline(name='p3b_tp_'+hashlib.sha256(TP_CUDA.encode()).hexdigest()[:12],cpp_sources=TP_CPP,cuda_sources=TP_CUDA,functions=fs,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_120,code=sm_120'])
 mod.br_init();NF_OPTS['pmz']=mod.pmz_init();[mod.oc_info(n,f,1) for n in (64,128,192,256) for f in (0,1)];[mod.tp_info(f,s) for f in (0,1) for s in (0,1)];OP_MODS.extend([mod]*4)
def op_plan(data,bank,arm):
 global OP_ARM
 OP_ARM=arm
 if arm==0:return nf_light_plan(data,bank['base'],bank['native'],bank['streams'][0],4)
 if arm==1:return reuse_plan(data,bank['base'],bank['native'],bank['streams'][1],3)
 return tp_plan(data,bank['base'],bank['native'],bank['streams'][arm],4)
def op_prepare(w):
 w=dict(w,post_norm_w=w['post_expert_norm']);base=nf_perm(nf_wide(m128_hh_issue_prepare(nf_ilv(w))));native=m128_hh_issue_nt_prepare(w)
 return dict(base=base,native=native,streams=[([torch.cuda.Stream() for _ in range(5)],torch.cuda.Event(),[torch.cuda.Event() for _ in range(5)]) if a==1 else (torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()) for a in range(4)])
'''+rt
 compile(body,'<tokpair>','exec');(H/'tokpair_payload.py').write_text(body);(H/'tokpair.cu').write_text(cu)
 for arm,name in ((2,'rs'),(3,'ss')):
  adapter='# okc: problem=muse_moe_w4a16\ndef lab_cold_l2():pass\n'+body+f'\nSCREEN_ARM={arm}\n'+'''def prepare(weights):
 op_build()
 w={k:(v[:32].contiguous() if k.startswith(('w13','w2')) else v) for k,v in weights.items()};w['post_expert_norm']=w['post_norm_w'];return op_prepare(w)
def kernel_fn(data,state):return op_call(data,state,SCREEN_ARM)
'''
  (H/f'screen_{name}.py').write_text(adapter)
 manifest=dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),CTA2_prefix_sha256=hashlib.sha256(ctrl.encode()).hexdigest(),arms=['S14','CTA2','RS_RAW_SHARE','SS_DECODE_SHARE'])
 (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)
if __name__=='__main__':main()
