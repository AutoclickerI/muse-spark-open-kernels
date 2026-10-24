"""Build glue-only candidate from the pinned, immutable best5."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).parent
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best5.py'
OUT=BASE.with_name('cand_r3_best5_glue.py')
SHA='a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150'
s=BASE.read_text();assert hashlib.sha256(BASE.read_bytes()).hexdigest()==SHA
# Delete old boundary/routing kernels and their atomic rendezvous helper.
a=s.index('@triton.jit\ndef _group_barrier');b=s.index('# Post-expert norm and weighted sum:',a)
a=s.rfind('# --------------------------------------------------------------------------------------------',0,a)
s=s[:a]+(HERE/'kernels.py.inc').read_text()+'\n\n# --------------------------------------------------------------------------------------------\n'+s[b:]
# Preserve all compute calls, selecting only regular projections. The old PREFIX
# changes O/pool GEMMs; this task explicitly excludes that mechanism.
a=s.index('    sync=(torch.empty if _chunks(t)==1 else torch.zeros)',s.index('def kernel_fn'))
b=s.index('    qkv=project',a)
s=s[:a]+s[b:]
a=s.index("    prefix,pdl=state['prefix_plan']",s.index('def kernel_fn'));b=s.index('    moe=experts_v9',a)
s=s[:a]+'''    a=project(attn,state['o'],'o',state)
    h,stream1=boundary(a,data['stream'],state['alpha1'],state['beta1'],state['pre_ffn_w'])
    pr=project(h,state['pool_router'],'pool_router',state,out_dtype=torch.float32)
    x,ids,wts,lids=route_and_norm(pr,state,data)
'''+s[b:]
s=s.replace("                           (sf[2],si[2]),postw=state['post_ffn_w'],use_pdl=state['pdl_plan'].get((t,'b2'),True))", "                           postw=state['post_ffn_w'])",1)
# Retain compatible glue helper entry points for the existing projection tuner,
# but do not compile or time old GEMM-fusion alternatives.
a=s.index("    state['pdl_plan']=_tune_pdl",s.index('def prepare'));b=s.index('    torch.cuda.synchronize(device)',a)
s=s[:a]+'''    state['pdl_plan']={(t,p):True for t in _TUNE_TOKENS for p in ('b1','b2')}
    state['prefix_plan']={t:('regular',True) for t in _TUNE_TOKENS}
    phases['pdl_prefix']=time.monotonic()-started-sum(phases.values())
'''+s[b:]
# Unused old prefix functions would require keeping the old kernels. Delete the
# entire obsolete prefix region and its tuner; no weight/compute source changes.
a=s.index('def _tune_pdl(');b=s.index('ATTN_PORT_CUDA_SRC = ',a)
s=s[:a]+s[b:]
a=s.index('def _tune_prefix(');b=s.index('def stage_attention',a)
s=s[:a]+s[b:]
s=s.replace('"""V9 decode layer with receipt-guided cost and attention portfolios.', '"""Best5 with self-contained PDL residual/norm and routing glue.')
# QK glue prologue reads only input metadata and static scales, before the
# dependency wait on QKV. Attention math and all compute kernels stay unchanged.
s=s.replace('    _pdl_start(USE_PDL)\n    _pdl_wait(USE_PDL)\n    W: tl.constexpr',
'''    _pdl_start(USE_PDL)
    cache_row=tl.load(LOC+t)
    kscale=tl.load(KSCALE);vscale=tl.load(VSCALE)
    if ROPE:
        half=tl.arange(0,D//2)
        angle=tl.load(POS+t).to(tl.float32)*tl.load(INV_FREQ+half)
        rope_cos=tl.cos(angle)[None,:];rope_sin=tl.sin(angle)[None,:]
    _pdl_wait(USE_PDL)
    W: tl.constexpr''',1)
s=s.replace('        pos = tl.load(POS + t).to(tl.float32)\n        half = tl.arange(0, D // 2)\n        ang = pos * tl.load(INV_FREQ + half)\n        c = tl.cos(ang)[None, :]\n        sn = tl.sin(ang)[None, :]',
'        c=rope_cos;sn=rope_sin',1)
s=s.replace('    row = tl.load(LOC + t)\n','    row = cache_row\n',1)
s=s.replace('(k / tl.load(KSCALE)).to(tl.float8e4nv)', '(k / kscale).to(tl.float8e4nv)',1)
s=s.replace('(v / tl.load(VSCALE)).to(tl.float8e4nv)', '(v / vscale).to(tl.float8e4nv)',1)
s=s.replace("    phases['pdl_prefix']=time.monotonic()-started-sum(phases.values())",
"    _tune_glue(state,flush,device)\n    phases['pdl_prefix']=time.monotonic()-started-sum(phases.values())",1)
# The projection tuner needs no scratch now that the glue has no counters.
s=s.replace('    sync = (torch.zeros((t, 4), device=x.device), torch.zeros((t,), device=x.device, dtype=torch.int32))',
            '    sync = None',1)
s+='\n\n'+(HERE/'tuner.py.inc').read_text()
OUT.write_text(s)
print(OUT.relative_to(ROOT),hashlib.sha256(OUT.read_bytes()).hexdigest())
