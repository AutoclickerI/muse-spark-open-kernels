"""S14 decode-1 glue chains: retain independent operands across the dependency wait."""
import ast,hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
sp=importlib.util.spec_from_file_location('d1_previous',HERE.parent/'pdl_audit/patch_d1_s14.py');previous=importlib.util.module_from_spec(sp);sp.loader.exec_module(previous)
MODES=('weights','late_prew','hints','stream','c4')

def source(base,mode):
 assert mode in MODES
 cu=previous.cuda_source(base)
 if mode=='late_prew':
  # The final normalization weight is consumed only after both reductions.
  cu=cu.replace('    w[g] = gp_load4(prew + j);','',1)
  anchor='    s[g] = *reinterpret_cast<const float4*>(stream + (long long)t * HID + j);'
  cu=cu.replace(anchor,'    w[g]=gp_load4(prew+j);\n'+anchor,1)
  line='    w[g] = q < QT ? gp_load4(prew + (rank * QT + q) * 4) : make_float4(0.f, 0.f, 0.f, 0.f);'
  assert line in cu;cu=cu.replace(line,'',1)
  anchor='    p[g] = q < QT ? sum_splits'
  cu=cu.replace(anchor,'    w[g] = q<QT ? gp_load4(prew+(rank*QT+q)*4) : make_float4(0.f,0.f,0.f,0.f);\n'+anchor,1)
 elif mode=='hints':
  # Hints fill L2 without making the weight values live across the wait.
  original=ast.literal_eval(next(n.value for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='GLUE_CL_CUDA'))
  cu=original.replace('namespace cg = cooperative_groups;', '''__device__ __forceinline__ void dc_hint(const void* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p) : "memory");
}
namespace cg = cooperative_groups;''')
  begin=cu.index('#pragma unroll\n  for (int g = 0; g < G; ++g)',cu.index('float4 a[G]'))
  end=cu.index('  pdl_wait();',begin);loads=cu[begin:end]
  hintloop='#pragma unroll\n  for(int g=0;g<G;++g){\n    const int j=(rank*NT*G+g*NT+tid)*4;\n    if((tid&7)==0){dc_hint(alpha+j);dc_hint(beta+j);dc_hint(prew+j);if(postw)dc_hint(postw+j);}\n  }\n'
  cu=cu[:begin]+hintloop+'  pdl_wait();\n'+loads+cu[end+len('  pdl_wait();\n'):]
  begin=cu.index('#pragma unroll\n  for (int g = 0; g < GQ; ++g)',cu.index('float4 w[GQ]'))
  end=cu.index('  pdl_wait();',begin);loads=cu[begin:end]
  hintloop='#pragma unroll\n  for(int g=0;g<GQ;++g){const int q=g*NT+tid;if(q<QT && (tid&7)==0)dc_hint(prew+(rank*QT+q)*4);}\n  if(tid==0){dc_hint(bias+rank*EP);dc_hint(lstart);}\n'
  cu=cu[:begin]+hintloop+'  pdl_wait();\n'+loads+cu[end+len('  pdl_wait();\n'):]
 elif mode=='stream':
  # Only b1 receives an original timed-call residual. b2's b1-produced residual
  # remains after wait: early-release chains do not establish its readiness.
  cu=cu.replace('float4 a[G], b[G], w[G], q[G];','float4 a[G], b[G], w[G], q[G], s[G];',1)
  cu=cu.replace('    a[g] = gp_load4(alpha + j);','    if(!postw)s[g]=gp_load4(stream+(long long)t*HID+j);\n    a[g] = gp_load4(alpha + j);',1)
  cu=cu.replace('  float4 s[G], br[G];','  float4 br[G];',1)
  cu=cu.replace('    s[g] = *reinterpret_cast<const float4*>(stream + (long long)t * HID + j);','    if(postw)s[g] = *reinterpret_cast<const float4*>(stream + (long long)t * HID + j);',1)
 return cu

def patch(base,mode):
 assert hashlib.sha256(base.encode()).hexdigest()==BASE_SHA
 cu=source(base,mode);rows=tuple((1,1,w,r) for w in (2048,0) for r in ('b1','route','b2'))
 suffix=previous.RUNTIME.replace('__GP_CUDA__',repr(cu),1).replace('__GP_ROWS__',repr(rows),1)
 # Retain exact prior wrapper API while making row selection sufficient in the lab.
 if mode=='c4':
  suffix=suffix.replace("p,s=branch;t=stream.shape[0];_,c,nt,rounds=geo;post=phase=='b2'", "p,s=branch;t=stream.shape[0];_,c,nt,rounds=geo;c=4;post=phase=='b2'")
  suffix=suffix.replace('p,s=pr;t=p.shape[1];dev=p.device;_,c,nt,_=geo','p,s=pr;t=p.shape[1];dev=p.device;_,c,nt,_=geo;c=4')
 base=base.replace('    _gate_split_prepare(state,device)\n','    _gate_split_prepare(state,device)\n    gp_module()\n',1)
 base=base.replace('    _best7_prepare_settings(state,weights,device)\n',"    _best7_prepare_settings(state,weights,device)\n    for key,plan in state['best7_plans'].items():plan['gp_key']=key\n",1)
 return base+'\n\n'+suffix

if __name__=='__main__':
 b=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
 for mode in MODES:
  s=patch(b,mode);(HERE/(mode+'_trial.py')).write_text(s);print(mode,hashlib.sha256(s.encode()).hexdigest())
