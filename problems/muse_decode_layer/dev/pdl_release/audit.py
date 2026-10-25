"""Source-backed critical-path release/wait inventory for frozen S15."""
import ast
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[1]
s=(P/'cand_r3_best7_s15.py').read_text();tree=ast.parse(s)
assert hashlib.sha256(s.encode()).hexdigest()=='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
L={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
   and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
rows=[]

def row(kernel,current,successor,wait,launch,source,detail):
    rows.append(dict(kernel=kernel,current=current,successor=successor,successor_wait=wait,
                     successor_launch=launch,source=source,detail=detail))

row('NF qkv3/qkv_gate warp and cluster GEMVs','entry','partial-port / last-CTA or qkv sumcast','yes','PDL',
    '_port_cuda','PDL inserted at the first shared declaration; wait before X read; selected T1/32 qkv roles')
row('balanced _gb_kernel / generic _gemv_kernel / pool bal2 _gi_kernel','entry','qkv sumcast / route_cl / boundary','yes','PDL',
    '_pdl_start / _gi_kernel','release precedes math; wait before activations')
row('coded_gemv V3 pool_router','entry','route_cl','yes','PDL','CD_CUDA',
    'weights/escape metadata can issue before wait; activations wait; T1 qkv_gate coded cell is bypassed by gate split')
row('NF O / NF full-unpooler warp, OL, MW','entry','boundary_cl / _boundary_fast / _boundary_late','yes','PDL',
    '_port_cuda / _nf_pdl_wide','producer release and predecessor wait are inserted by port; chosen T1/8/16/32 O and T16/32 unpooler')
row('VDSK _vdsk_gemm qkv M128','implicit exit','vq8_main','yes','PDL','_vdsk_gemm',
    'no explicit release; layer input X and W do not depend on an earlier in-layer producer; E adds entry release')
row('cuBLAS / cuBLASLt projection main and optional split-K reduce','implicit exit (external)','qk prep / VQ8 / boundary','yes for owned consumers','PDL after library completion',
    'gemv / _project_variant / PG_RAW','library body is not owned; no assumption about proprietary internal trigger instructions; S15 does not request early consumer execution inside the library')
row('_attn_sumcast_kernel','entry','attn_main portfolio / portable main','NO wait in portfolio S15; ordinary launch serializes','ordinary',
    '_attn_sumcast_kernel / ATTN_PORT_CUDA_SRC','sumcast has predecessor wait; next main normally launches, so input is safe. E adds wait before qkv load + PDL launch to overlap its independent metadata prologue')
row('attn_main portfolio q1','entry','gate split / NF O','yes','PDL','ATTN_PORT_CUDA_SRC',
    'one clustered grid includes merge; successor gate computes X/W independently then waits before Y, or O waits before activation')
row('partial-port full decode1','entry','gate split','yes (only last split CTA reads Y)','PDL','PARTIAL_CUDA',
    'partial projection input read follows griddepcontrol.wait; historical cache reads precede it')
row('last-CTA sliding decode1','entry','gate split','yes (only last split CTA reads Y)','PDL','cs_CUDA',
    'prep_decode waits before projection input; counter-reset is part of producer completion')
row('gate split projection / gate epilogue','entry','NF O / original O','yes','PDL','_gate_split_kernel',
    'X is original layer input; after whole gate GEMV last CTA waits before attention Y read')
row('vq8_main verify16/32 full','entry','vq8_combine','yes','PDL','VQ8_CUDA',
    'consumer warps wait before qkv; independent cache gather may run before wait; main releases on every thread')
row('vq8_combine','entry','O projection','library completes serially; owned variants wait','ordinary library or PDL native','VQ8_CUDA',
    'reads all main partials after wait; large-M O is Lt or cuBLAS at prepared choice')
row('_qk_prep_kernel / _best7_old_qk_prep_kernel','entry','FA3 or portable attention','external FA3 serialized; portable split waits','ordinary FA3 / PDL Triton',
    'qk_prep / _attention_fa3_public','qkv producer output read after wait; new KV writes stay ordered to FA3')
row('FA3 attention main and internal combine','implicit exit (external)','BF16 cast / _norm_gate_kernel','cast is ordinary; norm waits','ordinary cast / PDL norm',
    '_attention_fa3_public','compiled library boundary; cannot infer its proprietary/binary PDL internals from S15 source')
row('torch BF16 cast after FA3','implicit exit (library pointwise)','_norm_gate_kernel','yes','PDL','_attention_fa3_public',
    'only executes if FA3 output dtype requires conversion; ordinary library dispatch')
row('_norm_gate_kernel','entry','NF O / cuBLAS or Lt O','owned NF waits; libraries serialize','PDL native / ordinary library',
    '_norm_gate_kernel','output normalization reads attention only after wait')
row('portable attention CUDA attn_main','entry','portable attn_combine','yes','PDL','CUDA_SRC',
    'main lacks input wait and launches ordinarily; scored H200 uses portfolio except direct-partial and FA3 rows')
row('portable attention CUDA attn_combine','entry','gate / O','yes for owned consumers','PDL','CUDA_SRC',
    'wait is at entry before reading partials')
row('portable Triton _attn_split_kernel / _attn_combine_kernel','entry','combine / O','yes','PDL',
    '_attn_split_kernel / _attn_combine_kernel','local ordinary mode verifies this path; not scored native full-verify implementation')
row('boundary_cl b1/b2','entry','pool/router / return','pool/router waits; return has graph completion','PDL or none','GLUE_CL_CUDA',
    'weight vectors can preload before wait; branch and stream reads follow wait; final b2 release has no in-layer successor')
row('route_cl','entry','moe_var / nf route_live','moe_var yes; route_live NO, ordinarily launched','PDL small-M / ordinary large-M',
    'GLUE_CL_CUDA / NF_CUDA','large-M wrapper has a full completion boundary; E enables route_live PDL + wait')
row('_boundary_fast / _boundary_late / _route_fast and best5 glue','entry','projection / experts / return','owned PDL consumers wait','PDL',
    '_pdl_start','all load dependent branch values after _pdl_wait; scored M128/256 use row-per-CTA geometry')
row('moe_var flags95/69','implicit exit','unpooler slice / NF unpooler','yes','PDL','SMALLM_VAR_CUDA',
    'PDL flag currently waits on router but never releases its successor; E adds entry release, counters and finalizer unchanged')
row('V9 recoded/NT2 generic moe_fused','implicit exit','unpooler','owned consumers wait; library serializes','PDL owned / ordinary library',
    'V9_CUDA_SRC / V9_NT2_CUDA_SRC','nearby other-plan only; ordinary input launch guarantees routing complete; E entry release on recoded NT1 source')
row('N_full route_live','implicit exit','fork events then heavy/light/pmz W13','event completion','ordinary launches','NF_CUDA',
    'E adds entry release and predecessor wait; ordinary record/wait events remain completion boundaries')
row('N_full heavy and light W13 gemm','implicit exit','same-stream W2 gemm','NO S15 explicit wait; ordinary serialization','ordinary','NF_CUDA',
    'E adds entry release, PDL launch, and wait before TASK/activation reads; fork and join edges unchanged')
row('N_full heavy and light W2 gemm','implicit exit','join events -> finish','event completion','ordinary + events','NF_CUDA',
    'E adds entry release; no early event signalling; all produced Y/SS visible before join')
row('N_full small-expert phase W13','implicit exit','same-stream phase W2','NO S15 explicit wait; ordinary serialization','ordinary','NF_CUDA',
    'E entry release + PDL launch; W2 waits before counts/ACT; synthetic tuning work is not cached')
row('N_full small-expert phase W2','implicit exit','side-stream joins then finish','event completion','ordinary + events','NF_CUDA',
    'E entry release and wait; joins still wait for full-grid completion')
row('N_full finish<4>','implicit exit','unpooler Lt','library serialization; no owned wait inside Lt','ordinary library','NF_CUDA',
    'E entry release + consumer wait for prior phase; Lt remains a completion boundary')

# Call order is stage-exact; subkernel expansion references all rows above. Native
# tuner choice for external Lt is reported as a family rather than guessed kernel SASS.
loads=[]
for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
 for w in (2048,0):
  t=b*l;split=l==1 and (b,bool(w)) in {(1,True),(1,False),(16,True),(32,False)}
  attention=('last-CTA sliding' if w else 'direct-partial full') if b==l==1 else 'attn_main' if l==1 else 'vq8 main+combine' if b>1 and not w else 'qk prep+FA3+norm_gate'
  loads.append(dict(batch=b,q_len=l,window=w,stages=[
   'NF qkv3' if split and t in (1,32) else 'balanced qkv3' if split else 'NF qkv_gate' if t==32 else 'VDSK qkv' if t==128 else 'cuBLASLt qkv' if t==256 else 'balanced qkv',
   'qkv sumcast' if l==1 and b>1 else 'direct qkv partials / BF16',attention,
   'gate_split' if split else 'gate fused into attention','NF O' if t<=32 else 'Lt/cuBLAS O',
   'cluster b1' if t<=32 else 'Triton b1','coded pool/router' if t==1 else 'bal2 pool/router' if t==32 else 'balanced pool/router' if t<=32 else 'Lt/cuBLAS pool/router',
   'cluster route' if t<=32 else 'Triton route','moe_var' if t<=32 else 'N_full route+3 branches+finish',
   'NF unpooler' if t in (16,32) else 'generic slice GEMV' if t<=8 else 'Lt unpooler',
   'cluster b2' if t<=32 else 'Triton b2']))
result=dict(candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),rows=rows,shapes=loads,
            semantics_source='https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html',
            library_limit='implicit exit means S15 requests only ordinary completion semantics; external internal SASS is not inferred')
(H/'trigger_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(len(rows),'families,',len(loads),'scored shape paths')
