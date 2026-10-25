"""S7N: S7R with the M=128 and M=256 MoE on N_full (MoE-host labs c01b1916e2c74af8, 14f942d57e764ebf).

N_full runs on H_H's u8 base bank and the recoded NT1 bank that S7 already prepares for M=128: a one-CTA live
route that also writes the heavy-only (count>16) 32-row task list, then one fork: side = N32 WGMMA on the heavy
experts, W13 then W2, launched first; main = NT1 for experts with <=16 tokens, W13 (zeroing their norm rows) then
W2; joined before the distributed finalizer (stream assignment from 880c3881: N_swap 0.9799 / 0.9882 vs NT1 first). M=128: 207.40 vs S7's H_H 234.78 us (paired 0.8842, bit-equal); M=256: 298.68 vs
mixed_parallel 342.80 us (0.8701), R_full 316.70 (0.9230) in the same run; 12 seeds x3 fixtures, all TASK PASS. mixed_parallel's engine (FX/WI/rc repack, its bank and
its native compile) and the R_full cell are deleted: nothing else used them.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
BASE = R / 'problems/muse_decode_layer/cand_r3_best7_s7r.py'
BASE_SHA = '56c51fa026210357f9865d114e6bf7e609c52049e40b263f589d23ade95a8101'
BR = R / 'problems/muse_moe_w4a16/dev/mixed_wgmma/branches'
sys.path[:0] = [str(BR), str(R / 'tools')]
import sources  # noqa: E402
from okc_run import rejected_words  # noqa: E402

KINDS = ((20, 32, 1, 2, 0, 0, 0), (21, 32, 0, 2, 0, 0, 0))  # W13 / W2 N32, lean K loop, on the heavy-only T32 list
DEAD = ('pm_plan', 'pm_module', 'S5_HH_CUDA', 'S5_HH_CPP', 'S5_HH_FUNCTIONS', 'ISSUE_OPTS', 'FX_CUDA', 'FX_CPP', 'FX_CONFIGS', 'WI_CPP', 'fx_module', 'fx_prepare', 'fx_from_packed', 'fx_plan', 'wi_prepare',
        'wi_plan', 'wi_call', 'pm_call', 'RF_CUDA', 'RF_CPP', 'rf_module', 'rf_plan', 'rf_call', 'rc_masks', 'RC_SIGNS', 'RC_ZEROS', 'rc_repack')
COMMENTS = ('# H200 770809cf159f41e2: exact H_H expert/phase mix at M128.\n','# H200 8483c0c6783a4df1: exact M256 mixed_parallel policy6/grid2112.\n',
            '# MoE lane H200 110599ba6b6144d4: M256 R_full on mixed_parallel\'s buffers (bit-equal; paired 0.9227).\n')
CELL = '''

# MoE lane H200 c01b1916e2c74af8 / 14f942d57e764ebf / 880c388117ed4e16: N_full at M=128 (paired 0.8842 vs S7's H_H,
# bit-equal) and M=256 (0.8701 vs mixed_parallel); heavy on the side stream, launched first: a further 0.9799 / 0.9882;
# heavy kernel's lean K loop (858ff2fa): a further 0.9613 / 0.9426.
NF_CUDA={cuda!r}
NF_CPP={cpp!r}
NF_OPTS={{}}
@functools.lru_cache(None)
def nf_module():
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='nf_m256_'+hashlib.sha256(NF_CUDA.encode()).hexdigest()[:12],cpp_sources=NF_CPP,cuda_sources=NF_CUDA,
        functions=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run'],
        no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    mod.br_init();NF_OPTS['pmz']=mod.pmz_init()
    return mod

def nf_plan(data,base,native,streams):
    x=data['hidden_states'];m=x.shape[0];dev=x.device;cap=m*8+32*128;mod=nf_module()
    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c)
    tasks=[torch.empty_like(off) for _ in range(4)];slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    ids=data['topk_ids'].int().contiguous();tw=data['topk_weights'].float().contiguous();pos=torch.empty_like(ids)
    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev)
    out=torch.empty_like(x);start=data['config'].get('local_start',0)
    h13=[base['w13'],base['w13_sf'],base['w13_gs'],x,c,off,order,tasks[1],slots,base['pnw'],act,y,ss]
    h2=[base['w2'],base['w2_sf'],base['w2_gs'],act,c,off,order,tasks[1],slots,base['pnw'],act,y,ss]
    nv=[x,base['pnw'],native['w13p'],native['w2p'],base['w13_gs'],base['w2_gs'],c,off,slots,act,y,ss]
    p13,p2,pn=([t.data_ptr() for t in v] for v in (h13,h2,nv))
    rp=[t.data_ptr() for t in (ids,c,off,order,tasks[0],slots,pos,tasks[1],tasks[2],tasks[3])]
    fp=[t.data_ptr() for t in (off,ids,tw,pos,y,ss,out)]
    owners=(h13,h2,nv,ids,tw,pos,tasks)
    stages=2
    while stages>1 and mod.pmz_smem(stages,m)>NF_OPTS['pmz']:stages-=1
    sms=torch.cuda.get_device_properties(x.device).multi_processor_count
    grids=[sms*mod.pmz_resident(stages,m,f) for f in (0,1)]
    side,fork,join=streams;grid=1056 if m<=128 else 2112  # heavy-branch grids the H200 labs timed
    def route():
        _=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)
    def body():
        _=owners;parent=torch.cuda.current_stream()
        fork.record(parent);side.wait_event(fork)
        with torch.cuda.stream(side):  # heavy first: launched ahead of NT1 it takes the SMs it needs (880c3881 N_swap)
            mod.br_run(p13,20,grid,m,side.cuda_stream);mod.br_run(p2,21,grid,m,side.cuda_stream);join.record()
        mod.pmz_run(pn,1,16,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,16,m,stages,grids[0],parent.cuda_stream)
        parent.wait_event(join)
    def finish():
        _=owners;mod.br_finish(fp,m,start,torch.cuda.current_stream().cuda_stream);return out
    return dict(steps=(route,body,finish),out=out,owners=owners)

def nf_call(data,state):
    for f in nf_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'])['steps']:out=f()
    return out
'''
CELL_HEAD = ('"""M=128/256 N_full drop-in cell (H200 c01b1916, 14f942d5): paste into a best7 candidate after _s4_prepare_banks.\n'
             "Needs the host's functools, hashlib, torch and load_inline; state['s5_hh_base'] (H_H u8 base bank),\n"
             "state['moe_bank'] (recoded NT1 bank) and state['nf_streams']=(torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event())\n"
             'made in prepare. Call nf_call(data, state) at M=128 and M=256. dev/s7n/build.py shows the full integration on S7R."""\n')
ANCHOR = '\ndef _s4_gate_moe('
PY_EDITS = (
    ("    if name=='hh_m128':return pm_call(data,state['s5_hh_base'],state['moe_bank'],'H','H')",
     "    if name=='hh_m128':return nf_call(data,state)", 1),
    ("    if name=='mixed_parallel_m256':return rf_call(data,state['s6_bank'])",
     "    if name=='mixed_parallel_m256':return nf_call(data,state)", 1),
    ("    graph=_best5_graph(lambda data:rf_call(data,state['s6_bank']),fixtures[0])",
     "    graph=_best5_graph(lambda data:nf_call(data,state),fixtures[0])", 1),
    ("bank='lossless recoded W13 and W2',schedule='R_full: <=16-row groups on N16 W13->W2 beside N32 groups, forked once, joined before the finalizer')",
     "bank='H_H u8 base + recoded NT1 bank',schedule='N_full: NT1<=16 W13->W2 beside heavy N32 W13->W2, forked once, joined before the finalizer')", 1),
    ("        s6_future=native_pool.submit(fx_module)\n", "        nf_future=native_pool.submit(nf_module)\n", 1),
    ("        s5_large_future=native_pool.submit(pm_module)\n", "", 1),
    ("        state['s5_large_mod']=s5_large_future.result()\n", "", 1),
    ("        state['s6_mod']=s6_future.result()\n", "        nf_future.result()\n", 1),
    ("    s6_started=time.monotonic()\n    state['s6_bank']=wi_prepare(weights)\n    torch.cuda.synchronize(device)\n    phases['s6_repack']=time.monotonic()-s6_started\n",
     "    state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event())\n", 1),
)


PATCH_START = '# MoE lane H200 c01b1916e2c74af8'
PATCH_END = "def nf_call(data,state):\n    for f in nf_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'])['steps']:out=f()\n    return out\n"
PATCH = """\"\"\"patch(src) -> src: replace the N_full cell block of a best7 candidate (S7N/S10 onward) with the current cell.
Written by dev/s7n/build.py together with branches/nf_cell.py; the block runs from the MoE-lane comment to nf_call.\"\"\"
BODY = {body}
START = {start}
END = {end}


def patch(src):
    i = src.index(START)
    assert src.count(START) == 1 and src.count(END) == 1
    j = src.index(END) + len(END)
    assert i < j
    return src[:i] + BODY + src[j:]
"""


def main():
    s = BASE.read_text()
    assert hashlib.sha256(s.encode()).hexdigest() == BASE_SHA
    for old, new, n in PY_EDITS:
        assert s.count(old) == n, (old, s.count(old))
        s = s.replace(old, new)
    for c in COMMENTS:
        assert s.count(c) == 1, c
        s = s.replace(c, '')
    tree = ast.parse(s)
    lines = s.splitlines(keepends=True)
    drop = set()
    for n in tree.body:
        names = [n.name] if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else \
            [t.id for t in getattr(n, 'targets', []) if isinstance(t, ast.Name)] + \
            [e.id for t in getattr(n, 'targets', []) if isinstance(t, ast.Tuple) for e in t.elts]
        if names and all(k in DEAD for k in names):
            first = min([n.lineno] + [d.lineno for d in getattr(n, 'decorator_list', [])])
            drop.update(range(first - 1, n.end_lineno))
    s = ''.join(l for i, l in enumerate(lines) if i not in drop)
    for k in DEAD:
        assert not re.search(r'\b' + k + r'\b', s), k
    for k in ('s6_bank', 's6_mod', 's6_future'):
        assert k not in s, k
    src = sources.sources()
    cuda = sources.kinds_macro(KINDS) + (BR / 'gemm.cu.inc').read_text() + '\n' + sources.pmz_source(sources.s6_text())
    cell = CELL.format(cuda=cuda, cpp=src['cpp_hh'])
    (BR / 'nf_cell.py').write_text(CELL_HEAD + cell.lstrip('\n'))
    body = cell.lstrip('\n')
    assert body.startswith(PATCH_START) and body.endswith(PATCH_END)
    (H / 'nf_patch.py').write_text(PATCH.format(body=repr(body), start=repr(PATCH_START), end=repr(PATCH_END)))
    assert s.count(ANCHOR) == 1
    s = s.replace(ANCHOR, cell + ANCHOR, 1)
    s = re.sub(r'\n{4,}', '\n\n\n', s)
    compile(s, 'cand', 'exec')
    assert not rejected_words(s), rejected_words(s)
    out = BASE.with_name('cand_r3_best7_s7n.py')
    out.write_text(s)
    m = dict(source=str(out.relative_to(R)), sha256=hashlib.sha256(s.encode()).hexdigest(), base=str(BASE.relative_to(R)), base_sha256=BASE_SHA,
             nf_cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(), deleted=list(DEAD),
             lab='c01b1916 M128 N_full 207.40 vs H_H 234.78 us (0.8842); 14f942d5 M256 N_full 298.68 vs mixed_parallel 342.80 us (0.8701); 880c3881 N_swap 0.9799 / 0.9882 vs N_full; 858ff2fa lean K loop L_swap 197.16 / 279.38 us, 0.9613 / 0.9426 vs N_swap',
             change='M128 and M256 MoE: live route + fork-once NT1<=16 (W13->W2) || heavy N32 (W13->W2) on the H_H u8 + NT1 banks; mixed_parallel engine deleted')
    (H / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    print(m['source'], m['sha256'], len(s))


if __name__ == '__main__':
    main()
