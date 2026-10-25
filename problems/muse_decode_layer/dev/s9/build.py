"""S9: S8 + BF16 qkv front (H200 87cd3a84) + small-M NT1 flags (H200 849e4017, cf1de160) + M=128/256 N_full
(H200 c01b1916, 14f942d5). Applied in order to a named base: each `--patch` module's patch(src) (default: the glue
lane's BF16 qkv front, `problems/muse_bf16_proj/dev/attn_front/patch_front_bf16.py`), then N_full, then small-M.

BF16 qkv front: attn_main rows and the layer ext get BF16 qkv from one PDL Triton sum+cast kernel instead of torch's
sum + cast (bit-exact; decode 1 sliding/full -4.21 / -4.46 us, decode 16 full -5.25, decode 16 sliding -1.52).

Small-M (`dev/nt1_timeline/smallm_var_cell.py`): the LOP3 NT1 kernel with compile-time flags replaces S4's
nt1_call on every nt1_lop3 row, same recoded bank. Router+MoE graph, 16 seeds paired vs INC: M=1 flags 95
(Z+SLOT+ZOUT+RSQ+NREG+PDL) -5.72 us (849e4017); M=8/16/32 flags 69 (Z+ZOUT+PDL) -4.11 / -4.83 / -4.46 us, ahead of
93 (+RSQ+NREG) at -3.40 / -4.55 / -3.76 in the same lab (cf1de160, read flush). PDL makes
the MoE a programmatic dependent of the route kernel: S8 routes every nt1_lop3 row with route_cl (releases its
dependents at entry) and launches nothing between route and MoE (asserted below). Only the flag words in the
map are instantiated (plus flags 0, which sizes the grid). S4's NT1 module and nt1_call are deleted.
M=128/256 (`dev/s7n/build.py`, MoE lane, applied as is): N_full on H_H's u8 base bank + the recoded NT1 bank;
M=128 207.40 vs H_H 234.78 us (paired 0.8842, bit-equal, c01b1916), M=256 0.8701 vs mixed_parallel and R_full 0.9230
in the same run (14f942d5). pm_call (H_H), R_full and mixed_parallel's engine and bank are deleted.

    python3 problems/muse_decode_layer/dev/s9/build.py [--base PATH] [--patch MODULE ...] [--flags 1:95,8:69,16:69,32:69]
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
D = R / 'problems/muse_decode_layer'
CELL = D / 'dev/nt1_timeline/smallm_var_cell.py'
PATCHES = [R / 'problems/muse_bf16_proj/dev/attn_front/patch_front_bf16.py']
spec = importlib.util.spec_from_file_location('s7n_build', D / 'dev/s7n/build.py')
s7n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s7n)  # puts the MoE branches dir and tools/ on sys.path
from okc_run import rejected_words  # noqa: E402
import sources  # noqa: E402

LABS = dict(smallm='849e40173ebd4bfc + cf1de1607d394778 (router+MoE, 16 seeds: M1 95 -5.72, M8/16/32 69 -4.11/-4.83/-4.46 us)',
            nfull='c01b1916e2c74af8 (M128 N_full 207.40 vs H_H 234.78 us, 0.8842) + 14f942d57e764ebf (M256 298.68 vs mixed_parallel 342.80 us, 0.8701)')
NT1_DEAD = ('S4_NT1_CUDA', 'NT1_CPP', 'NT1_MODULES', 'nt1_module', 'nt1_call')
SMALLM_EDITS = (
    ("        s4_small_future=native_pool.submit(nt1_module,'lop3')\n",
     "        s4_small_future=native_pool.submit(smallm_module)\n"),
    ("        state['s4_small_mod']=s4_small_future.result()\n",
     "        s4_small_future.result()\n"),
    ("    if name=='nt1_lop3':return nt1_call(data,state['s4_nt1_bank'],'lop3',2,264)\n",
     "    if name=='nt1_lop3':return smallm_experts(data,state['s4_nt1_bank'],SMALLM_FLAGS[data['hidden_states'].shape[0]])\n"),
    ("        counters={'lop3':torch.zeros(64,device=dev,dtype=torch.int32)},\n"
     "        banks={'lop3':dict(w13p=bank['w13p'],w2p=bank['w2p'])})\n",
     "        banks={'lop3':dict(w13p=bank['w13p'],w2p=bank['w2p'])})\n"
     "    smallm_bind(state['s4_nt1_bank'],SMALLM_FLAGS)\n"),
)


def sub(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)


def drop_defs(s, dead):
    """Delete the top-level statements that define only names in `dead`; assert no reference is left."""
    lines = s.splitlines(keepends=True)
    drop = set()
    for n in ast.parse(s).body:
        names = [n.name] if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else \
            [t.id for t in getattr(n, 'targets', []) if isinstance(t, ast.Name)] + \
            [e.id for t in getattr(n, 'targets', []) if isinstance(t, ast.Tuple) for e in t.elts]
        if names and all(k in dead for k in names):
            first = min([n.lineno] + [d.lineno for d in getattr(n, 'decorator_list', [])])
            drop.update(range(first - 1, n.end_lineno))
    s = ''.join(l for i, l in enumerate(lines) if i not in drop)
    for k in dead:
        assert not re.search(r'\b' + k + r'\b', s), k
    return s


def nfull(s):
    """The MoE lane's S7N transform (dev/s7n/build.py constants), applied to this base."""
    for old, new, n in s7n.PY_EDITS:
        assert s.count(old) == n, (old, s.count(old))
        s = s.replace(old, new)
    for c in s7n.COMMENTS:
        s = sub(s, c, '')
    s = drop_defs(s, s7n.DEAD)
    for k in ('s6_bank', 's6_mod', 's6_future'):
        assert k not in s, k
    src = sources.sources()
    cuda = sources.kinds_macro(s7n.KINDS) + (s7n.BR / 'gemm.cu.inc').read_text() + '\n' + sources.pmz_source(sources.s6_text())
    assert s.count(s7n.ANCHOR) == 1
    s = s.replace(s7n.ANCHOR, s7n.CELL.format(cuda=cuda, cpp=src['cpp_hh']) + s7n.ANCHOR, 1)
    return s, hashlib.sha256(cuda.encode()).hexdigest()


def smallm(s, flags):
    """Swap S4's NT1 for the flag-templated cell; instantiate only the mapped flag words."""
    cell = CELL.read_text()
    tree = ast.parse(cell)
    value = lambda key: next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == key)
    built, cuda = value('SMALLM_FLAGS_BUILT'), value('SMALLM_VAR_CUDA')
    words = sorted(set(flags.values()))
    assert set(words) <= set(built), (words, built)
    assert all(not f & 2 or m == 1 for m, f in flags.items()), 'SLOT is the M=1 dispatch'
    assert all(not f & 32 for f in words), 'META needs route_meta, which S8 does not pass'
    a = cuda.index('#define VAR_FLAGS(X) ')
    b = cuda.index('\n', a)
    cuda = cuda[:a] + '#define VAR_FLAGS(X) ' + ' '.join(f'X({f})' for f in words) + cuda[b:]
    code = f'\nSMALLM_VAR_CUDA = {cuda!r}' + cell[cell.index('\nSMALLM_VAR_CPP = '):]
    code = sub(code, f'SMALLM_FLAGS_BUILT = {built!r}', f'SMALLM_FLAGS_BUILT = {words!r}')
    block = ('\n# Small-M NT1 flags (dev/nt1_timeline/smallm_var_cell.py), H200 ' + LABS['smallm'] + '.\n'
             f'SMALLM_FLAGS = {flags!r}' + code + '\n\n')
    anchor = '\n# H200 dc3fea0a20a349bf: exact one-LOP3 small-M and distributed D32g M128.\nS4_NT1_CUDA='
    s = sub(s, anchor, block + anchor[anchor.index('\nS4_NT1_CUDA='):])
    for old, new in SMALLM_EDITS:
        s = sub(s, old, new)
    s = drop_defs(s, NT1_DEAD)
    return s, hashlib.sha256(cuda.encode()).hexdigest()


def check_pdl_rows(s, flags):
    """Every nt1_lop3 row: cluster route (released at entry) right before the MoE, no L2 prefetch fork."""
    tree = ast.parse(s)
    value = lambda key: next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == key)
    table = dict(value('BEST7_SETTINGS'))
    for key, glue in value('GLUE_CL_ROWS').items():
        table[key] = (glue,) + tuple(table[key][1:])
    rows = {k: v for k, v in table.items() if k != 'other' and v[2] == 'nt1_lop3'}
    assert {k[0] * k[1] for k in rows} == set(flags), (rows.keys(), flags)
    for key, (glue, _, _, projection) in rows.items():
        assert isinstance(glue, tuple) and glue[2][0] == 'cl', (key, glue)
        assert projection == 'auto' or not projection.get('l2_prefetch'), (key, projection)
    run = s[s.index('\ndef _best7_run('):s.index('\ndef kernel_fn(')]
    between = run[run.index('_best7_route(pr,state,data,plan,sync[1])'):run.index('moe=_best7_expert(')]
    assert between.count('\n') == 2 and "prefetch.begin('tail')" in between, between
    return sorted(rows)


def load(path):
    spec = importlib.util.spec_from_file_location('s9_patch_' + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=D / 'cand_r3_best7_s8.py')
    ap.add_argument('--patch', type=Path, nargs='*', default=PATCHES, help='modules with patch(src) -> src, applied in order')
    ap.add_argument('--flags', default='1:95,8:69,16:69,32:69')
    a = ap.parse_args()
    flags = {int(m): int(f) for m, f in (x.split(':') for x in a.flags.split(','))}
    base = a.base.resolve()
    s = base.read_text()
    base_sha = hashlib.sha256(s.encode()).hexdigest()
    patches = []
    for path in a.patch:
        path = path.resolve()
        s = load(path).patch(s)
        patches.append(dict(path=str(path.relative_to(R)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            out_sha256=hashlib.sha256(s.encode()).hexdigest()))
    head = '"""Best7 ('
    s = sub(s, head, head + f'S9 = {base.stem.split("_")[-1].upper()} + ' + ''.join(p['path'].rsplit('/', 1)[-1].removesuffix('.py') + ' + ' for p in patches)
            + f'small-M NT1 flags {flags} (H200 849e4017, cf1de160) + M128/M256 N_full (H200 c01b1916, 14f942d5); ')
    s, nf_sha = nfull(s)
    s, sm_sha = smallm(s, flags)
    rows = check_pdl_rows(s, flags)
    s = re.sub(r'\n{4,}', '\n\n\n', s)
    compile(s, 'cand', 'exec')
    assert not rejected_words(s), rejected_words(s)
    out = D / 'cand_r3_best7_s9.py'
    out.write_text(s)
    m = dict(source=str(out.relative_to(R)), sha256=hashlib.sha256(s.encode()).hexdigest(), base=str(base.relative_to(R)),
             base_sha256=base_sha, patches=patches, smallm_flags=flags, smallm_cuda_sha256=sm_sha, smallm_cell_sha256=hashlib.sha256(CELL.read_bytes()).hexdigest(),
             nf_cuda_sha256=nf_sha, pdl_rows=[list(k) for k in rows], deleted=list(s7n.DEAD) + list(NT1_DEAD), labs=LABS)
    (H / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    print(m['source'], m['sha256'], len(s))


if __name__ == '__main__':
    main()
