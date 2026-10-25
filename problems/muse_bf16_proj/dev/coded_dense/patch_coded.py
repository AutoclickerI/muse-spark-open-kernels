"""Coded dense GEMVs (P10) for a best7-family decode-layer candidate (S10, S11, ...): a patch module (patch(src) -> src,
as the integration lane's builders apply them) and a CLI.

    python3 problems/muse_bf16_proj/dev/coded_dense/patch_coded.py <cand.py> <out.py>
    python3 problems/muse_bf16_proj/dev/coded_dense/patch_coded.py --emit <module.py>   # frozen hand-off module

An emitted module carries coded.py's source as a string (BODY), so its SHA-256 pins the kernel too.

prepare() compiles the coded module in the native pool, encodes the roles listed in ROWS losslessly (coded.py: 12-bit code, per-16-row exponent window, escape list), checks the kernel's decode
bit for bit against every weight (raises on any difference), and routes each (tokens, role) in ROWS through
`_project_variant()` using state['coded_dense']: same (P [S, T, N] FP32, S) ABI, PDL release at entry, weights preloaded
before griddepcontrol.wait. ROWS come from the H200 lab (prof/lab/coded_gemv.py); the rows are forced into
proj_plan after the prepare tuners, and a decode-1 'stream' o override is dropped when ROWS codes o at t = 1.
The original BF16 weights stay (cuBLAS / LT / vdsk still serve M = 128 / 256).
S11 final keeps its gate split at decode 1, so full-qkv coding runs only at verify 1 (T=8).
The encoded banks live in state; dispatch survives the judge's tensor clones and address padding.
"""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (tokens, role) -> (NW, RT, PIPE, S, prew, V, C): the rows where the coded GEMV beat every BF16 control on the H200
# (320884442a6f4624, V3, 24 paired cold-L2 rounds): qkv t=1 10.74 vs bal 11.66, t=8 11.20 vs 11.78; pool_router t=1
# 18.75 vs 21.12, t=8 19.14 vs 21.33 us. o, the unpooler slice and t >= 16 stay on their BF16 GEMVs.
ROWS = {(1, 'qkv_gate'): (8, 1, 2, 22, 1, 3, 1), (8, 'qkv_gate'): (8, 1, 2, 22, 1, 3, 1),
        (1, 'pool_router'): (8, 1, 2, 11, 1, 3, 1), (8, 'pool_router'): (8, 1, 2, 11, 1, 3, 1)}


def _sub(src, old, new, count=1):
    if src.count(old) != count:
        raise SystemExit(f'expected {count} of {old[:70]!r}, found {src.count(old)}')
    return src.replace(old, new)


BODY = None  # an emitted module's frozen coded.py body


def coded_block(rows):
    body = BODY if BODY is not None else (HERE / 'coded.py').read_text().split('import torch.nn.functional as F\n', 1)[1]
    inst = sorted({(1 if t <= 8 else 2 if t <= 16 else 4, rt, pipe, v) for (t, _), (nw, rt, pipe, s, prew, v, c) in rows.items()})
    body = _sub(body, 'CD_INSTANCES = tuple((nt, rt, pipe, v) for v in (3, 4) for nt in (1, 2, 4) for rt in (1, 2) for pipe in (2, 4))',
                f'CD_INSTANCES = {tuple(inst)!r}')
    return ('\n\n# ---- P10 coded dense GEMVs (problems/muse_bf16_proj/dev/coded_dense/coded.py) ----\n'
            'import torch.nn.functional as F\n' + body +
            f'\n\n# (tokens, role) -> (NW, RT, PIPE, S, prew, V, C): H200 lab rows (prof/lab/coded_gemv.py)\nCD_ROWS = {rows!r}\n'
            '\n\ndef _cd_prepare(state):\n'
            '    """Store lossless codes in state, so judge clones carry the matching encoded weights."""\n'
            "    state['coded_dense'] = {}\n"
            '    for name in sorted({name for _, name in CD_ROWS}):\n'
            '        w = state[name]\n'
            '        c = cd_encode(w)\n'
            '        if not torch.equal(cd_decode(c).view(torch.int16), w.view(torch.int16)):\n'
            '            raise RuntimeError(f"coded {name}: decode differs from the weight")\n'
            "        state['coded_dense'][name] = c\n"
            '        print(f"CODED {name} {tuple(w.shape)} escapes={c[\'nesc\']} bytes={c[\'coded\'].numel()}", flush=True)\n\n\n'
            'def _cd_force(state):\n'
            '    for (t, name) in CD_ROWS:\n'
            "        state['proj_plan'][(t, name)] = 'custom'\n")



def patch(src, rows=None):
    rows = ROWS if rows is None else rows
    src = _sub(src, '\n_GEMV_CHOICES = {}\n', '\n_GEMV_CHOICES = {}\n' + coded_block(rows))
    src = _sub(src, 'def _project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):\n',
               'def _project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):\n'
               '    cfg = CD_ROWS.get((x.shape[0], name))\n'
               "    if cfg is not None:\n"
               '        NW, RT, PIPE, S, PREW, V, C = cfg\n'
               "        return cd_gemv(x, state['coded_dense'][name], S, NW=NW, RT=RT, PIPE=PIPE, prew=PREW,\n"
               '                       pdl=PDL, V=V, C=C)\n')
    src = re.sub(r'with ThreadPoolExecutor\(max_workers=(\d+)\) as native_pool:',
                 lambda m: f'with ThreadPoolExecutor(max_workers={int(m.group(1)) + 1}) as native_pool:\n'
                           '        cd_future=native_pool.submit(cd_module)', src, count=1)
    src = _sub(src, '        state=_prepare_weights(weights)\n',
               '        state=_prepare_weights(weights)\n        cd_future.result();_cd_prepare(state)\n')
    src = _sub(src, '    _tune_measured_lt(state,flush,device)\n',
               '    _tune_measured_lt(state,flush,device)\n    _cd_force(state)\n')
    if (1, 'o') in rows:  # the decode-1 rows' o 'stream' override would bypass the coded choice
        src = _sub(src, "{'o': ('stream', 6, 1)}", "{}", count=2)
    return src


def emit(out):
    """Write a self-contained copy of this module with coded.py's current body frozen in."""
    body = (HERE / 'coded.py').read_text().split('import torch.nn.functional as F\n', 1)[1]
    src = Path(__file__).read_text()
    src = _sub(src, '\nBODY = None  # an emitted module\'s frozen coded.py body\n',
               f'\nBODY = {body!r}  # frozen coded.py body (coded.py SHA-256 '
               f'{hashlib.sha256((HERE / "coded.py").read_bytes()).hexdigest()})\n')
    out.write_text(src)
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())


def main():
    if sys.argv[1] == '--emit':
        return emit(Path(sys.argv[2]))
    cand, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not ROWS:
        raise SystemExit('ROWS is empty: fill it from the H200 lab receipt first')
    src = patch(cand.read_text(), ROWS)
    out.write_text(src)
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
