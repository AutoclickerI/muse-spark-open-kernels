"""Embed the projection port in our sole owned decode candidate; no sibling files change."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
P=Path(__file__).resolve().parents[1];ROOT=P.parents[1];sys.path.insert(0,str(P/'dev'))
from lean_codegen import values_of,device_only
from build_portfolio_v2 import fused_source

base_path=ROOT/'problems/muse_decode_layer/cand_r3_v9_autotune_q1.py'
base=base_path.read_text();source=base
# The latest complete-layer receipt establishes V9 as the control. Avoid packing
# and compiling an unused second expert bank while tuning projections.
source=source.replace('        "moe_bank_v7": v7_prepare({**weights, "post_norm_w": weights["post_expert_norm"]}),\n','')
# Replace the old Torch-header Lt extension and all four tuning routines with the
# integrated, budgeted versions below. The decode lane's source stays untouched.
drop={'_lt_module','_measure_graph','_tune_projections','_tune_fa3','_tune_pdl','_tune_moe','_time_variant'}
drop_assign={'LT_CUDA_SRC','LT_CPP_SRC','_LT_MOD'}
lines=source.splitlines(keepends=True);spans=[]
for node in ast.parse(source).body:
    if isinstance(node,ast.FunctionDef) and node.name in drop:
        spans.append((node.lineno,node.end_lineno))
    elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in drop_assign for t in node.targets):spans.append((node.lineno,node.end_lineno))
for a,b in reversed(spans):lines[a-1:b]=[]
source=''.join(lines)
source=source[:source.rindex('\ndef prepare(weights):')]+source[source.index('\ndef experts_v4(',source.rindex('\ndef prepare(weights):')):]
# The partial tuple is deliberately preserved for every public projection path.
a=source.index('def _project_variant(');b=source.index('\ndef project(',a)
source=source[:a]+source[b:]

old=values_of((P/'cand_portfolio.py').read_text());skin=old['CUDA_SRC']
z=re.search(r'template <int NT, int TR, int U, int NA, int XP, int LD>\nvoid launch_t',skin).start()
skin=skin[:z]+'\n} // namespace muse_proj\n'
skin=skin.replace('#include <torch/extension.h>','').replace('#include <ATen/cuda/CUDAContext.h>','').replace('namespace cg = cooperative_groups;','')
skin=skin.replace('template <int NT, int TR, int U, int NA, int XP, int LD>','template <int NT, int TR, int U, int NA, int XP, int LD, bool OUTF32, bool MIX>')
skin=skin.replace('  constexpr bool OUTF32 = NA == 2;','')
skin=skin.replace('    int WN, int WK) {','    int WN, int WK, int LDOUT, int ZERO_TAIL) {')
needle='  const int tok0 = blockIdx.z * BT;'
skin=skin.replace(needle,needle+'''
  if constexpr(MIX) {
    if(row_cta>=4352) {
      if(blockIdx.y==0)for(int e=threadIdx.x;e<BT*BR;e+=blockDim.x){int t=e/BR,r=e%BR;if(tok0+t<M)reinterpret_cast<float*>(Out)[(size_t)(tok0+t)*LDOUT+row_cta+r]=0.f;}
      return;
    }
  }
  const bool use_lo = !MIX || row_cta>=4096;
''')
skin=skin.replace('  const __nv_bfloat16* wp1 = W1 + off;','  const __nv_bfloat16* wp1 = MIX ? (use_lo ? W1+off-(size_t)4096*K : W0+off) : W1+off;')
skin=skin.replace('for (int na = 0; na < NA; ++na)\n        for', 'for (int na = 0; na < NA; ++na)\n        for')
# Both the load and MMA loops have this exact per-row brace; the low matrix is
# structurally absent on pooler rows, while every router hi/lo value is consumed.
skin=skin.replace('for (int r = 0; r < TR; ++r) {\n          const __nv_bfloat16* p', 'for (int r = 0; r < TR; ++r) {\n          if constexpr(MIX) if(na==1 && !use_lo)continue;\n          const __nv_bfloat16* p')
skin=skin.replace('for (int j = 0; j < NT; ++j) {\n            const uint4& lo', 'for (int j = 0; j < NT; ++j) {\n            if constexpr(MIX) if(na==1 && !use_lo)continue;\n            const uint4& lo')
skin=skin.replace('(size_t)(tok0 + tok) * N + row_cta + row','(size_t)(tok0 + tok) * LDOUT + row_cta + row')
skin=skin.replace('      reinterpret_cast<float*>(Out)[idx] = v;','      {reinterpret_cast<float*>(Out)[idx] = v;if(ZERO_TAIL)reinterpret_cast<float*>(Out)[idx+N]=0.f;}')
specs=set()
for nt,tr,u,na,xp,ld in values_of((P/'cand_fused_select.py').read_text())['INSTANCES']:
    specs.add((nt,tr,u,na,xp,ld,na==2,False))
    if na==1:specs.add((nt,tr,u,na,xp,ld,True,False))
# Decode m=1/8 both use NT=1, and its 4608-row FP32 pool/router is
# independent of the standalone router's NA=2 specializations.
for (layer,t), (tr,u,wn,wk,ks,xp) in values_of((P/'cand_fused_select.py').read_text())['TUNED_CONFIGS'].items():
    nt=1 if t<=8 else 2 if t<=16 else 4 if t<=32 else 8
    specs.add((nt,tr,u,1,xp,0,False,False))
    specs.add((nt,tr,u,1,xp,0,True,False))
for nt,u,xp in ((1,4,1),(2,4,1),(4,4,0),(8,2,0)):
    specs.add((nt,1,u,2,xp,0,True,True));specs.add((nt,1,u,2,xp,0,True,False))
pair_configs=[(64,64,4,0,1,th) for th in (160,256)]
pair_configs += [(128,64,3,0,1,160),(128,64,4,0,1,256),(128,128,2,0,1,256)]
pair,ps=fused_source([(cfg,False) for cfg in pair_configs]+[((64,64,4,1,8,th),True) for th in (160,256)])
pair=device_only(pair,'muse_pair')
pair=pair.replace('int total) {','int total, int ldout, int zero_tail) {')
pair=pair.replace('(size_t)(tok0 + token) * N + row0 + row','(size_t)(tok0 + token) * ldout + row0 + row')
pair=pair.replace('if constexpr (OUTF32) reinterpret_cast<float*>(Out)[idx] = v;','if constexpr (OUTF32) {reinterpret_cast<float*>(Out)[idx] = v;if(zero_tail)reinterpret_cast<float*>(Out)[idx+N]=0.f;}')
host=(P/'dev/lean_host.cuh').read_text();host=host[host.index('static int lean_map('):host.index('struct LeanLayout{')]
host=host.replace('int mode,int split,int64_t work_elems','int mode,int split,int ldout,int zero_tail,int64_t work_elems')
host=host.replace('chunk,total);','chunk,total,ldout,zero_tail);')
host=host.replace('int mode,int split,int threads,','int mode,int split,int threads,int ldout,int zero_tail,')
host=host.replace('M,N,K,KW,mode,split,work_elems','M,N,K,KW,mode,split,ldout,zero_tail,work_elems')
host=host.replace('LEAN_PAIR_CASES',' '.join(f'LPAIR({bt},{bk},{st},{th},{str(of).lower()},{str(hh).lower()})' for bt,bk,st,th,of,hh in ps))
lt=(P/'dev/all_lt.cuh').read_text()
lt=lt.replace('int nn,int kind)', 'int nn,int kind,int ldout)')
lt=lt.replace('CUDA_R_16BF,n,m,n','CUDA_R_16BF,n,m,ldout').replace('LayoutCreate(&c,out,n,m,n)','LayoutCreate(&c,out,n,m,ldout)')
lt=lt.replace('int nn,int kind,size_t workspace','int nn,int kind,int ldout,size_t workspace')
lt=lt.replace('int nn,int kind,int index,int accumulate,','int nn,int kind,int ldout,int index,int accumulate,')
lt=lt.replace('stride,nn,kind);','stride,nn,kind,ldout);')
raw=(P/'dev/decode_proj_host.cuh').read_text().replace('DP_DEVICE',skin+pair).replace('DP_PAIR_HOST',host).replace('DP_LT_HOST',lt)
raw=raw.replace('DP_SKIN_CASES',' '.join('DP_SKIN('+','.join(str(v).lower() if isinstance(v,bool) else str(v) for v in c)+')' for c in sorted(specs)))
source+='\n'+(P/'dev/decode_proj_tuner.py').read_text()+"\n_DP_CUDA_SRC = r'''"+raw+"'''\n"
source+='DP_PAIR_CONFIGS = '+repr(tuple(pair_configs))+'\n'
source+='DP_SKIN_CONFIGS = '+repr(values_of((P/'cand_fused_select.py').read_text())['TUNED_CONFIGS'])+'\n'
source+='DP_PARENT_SHA256 = '+repr(hashlib.sha256(base.encode()).hexdigest())+'\n'
out=ROOT/'problems/muse_decode_layer/cand_r3_proj.py';compile(source,str(out),'exec')
if out.exists():raise FileExistsError('use a new filename for any already queued source')
out.write_text(source)
print(json.dumps({'source':str(out),'sha256':hashlib.sha256(source.encode()).hexdigest(),'parent_sha256':hashlib.sha256(base.encode()).hexdigest(),'skinny_specializations':len(specs),'pair_specializations':len(ps)},indent=2))
