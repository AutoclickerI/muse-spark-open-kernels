"""Coded dense GEMVs (P10) for a best7-family decode-layer candidate (S8, S9, ...).

    python3 problems/muse_bf16_proj/dev/coded_dense/patch_coded.py <cand.py> <out.py>

prepare() compiles the coded module in the native pool, encodes qkv_gate, o, pool_router, unpooler_slice and
unpooler losslessly (coded.py: 12-bit code, per-16-row exponent window, escape list), checks the kernel's decode
bit for bit against every weight (raises on any difference), and routes each (tokens, role) in ROWS through
`gemv()` as ('coded', NW, RT, PIPE, S): same (P [S, T, N] FP32, S) ABI, PDL release at entry, weights preloaded
before griddepcontrol.wait. ROWS come from the H200 lab (prof/lab/coded_gemv.py); the rows are forced into
proj_plan after the prepare tuners, and a decode-1 'stream' o override is dropped when ROWS codes o at t = 1.
The original BF16 weights stay (cuBLAS / LT / vdsk still serve M = 128 / 256).
"""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (tokens, role) -> (NW, RT, PIPE, S, prew). Filled from the H200 lab receipt.
ROWS = {}


def _sub(src, old, new, count=1):
    if src.count(old) != count:
        raise SystemExit(f'expected {count} of {old[:70]!r}, found {src.count(old)}')
    return src.replace(old, new)


def coded_block(rows):
    body = (HERE / 'coded.py').read_text().split('import torch.nn.functional as F\n', 1)[1]
    inst = sorted({(1 if t <= 8 else 2 if t <= 16 else 4, rt, pipe) for (t, _), (nw, rt, pipe, s, prew) in rows.items()})
    body = _sub(body, 'CD_INSTANCES = tuple((nt, rt, pipe) for nt in (1, 2, 4) for rt in (1, 2) for pipe in (2, 3, 4))',
                f'CD_INSTANCES = {tuple(inst)!r}')
    return ('\n\n# ---- P10 coded dense GEMVs (problems/muse_bf16_proj/dev/coded_dense/coded.py) ----\n'
            'import torch.nn.functional as F\n' + body +
            f'\n\n# (tokens, role) -> (NW, RT, PIPE, S, prew): H200 lab rows (prof/lab/coded_gemv.py)\nCD_ROWS = {rows!r}\n'
            '_CD_BANK = {}  # (data_ptr, N, K) -> coded weight: unpooler_slice shares the unpooler storage\n\n\n'
            'def _cd_prepare(state):\n'
            '    """Encode every role ROWS codes, check the kernel decode bit for bit, register the GEMV choices."""\n'
            '    for name in sorted({name for _, name in CD_ROWS}):\n'
            '        w = state[name]\n'
            '        c = cd_encode(w)\n'
            '        if not torch.equal(cd_decode(c).view(torch.int16), w.view(torch.int16)):\n'
            '            raise RuntimeError(f"coded {name}: decode differs from the weight")\n'
            '        _CD_BANK[(w.data_ptr(), *w.shape)] = c\n'
            '        print(f"CODED {name} {tuple(w.shape)} escapes={c[\'nesc\']} bytes={c[\'coded\'].numel()}", flush=True)\n\n\n'
            'def _cd_choices(state):\n'
            '    for (t, name), cfg in CD_ROWS.items():\n'
            '        n, k = state[name].shape\n'
            '        _GEMV_CHOICES[(t, n, k)] = (\'coded\',) + tuple(cfg)\n\n\n'
            'def _cd_force(state):\n'
            '    _cd_choices(state)\n'
            '    for (t, name) in CD_ROWS:\n'
            '        state[\'proj_plan\'][(t, name)] = \'custom\'\n')


def patch(src, rows):
    src = _sub(src, '\n_GEMV_CHOICES = {}\n', '\n_GEMV_CHOICES = {}\n' + coded_block(rows))
    src = _sub(src, '    choice = _GEMV_CHOICES.get((T, N, K))\n',
               '    choice = _GEMV_CHOICES.get((T, N, K))\n'
               "    if isinstance(choice, tuple) and choice[0] == 'coded':\n"
               '        _, NW, RT, PIPE, S, PREW = choice\n'
               '        return cd_gemv(x, _CD_BANK[(w.data_ptr(), N, K)], S, NW=NW, RT=RT, PIPE=PIPE, prew=PREW, pdl=PDL)\n')
    src = _sub(src, '            _GEMV_CHOICES[(t, n, k)] = choice\n',
               '            _GEMV_CHOICES[(t, n, k)] = choice\n    _cd_choices(state)\n')
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


def main():
    cand, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not ROWS:
        raise SystemExit('ROWS is empty: fill it from the H200 lab receipt first')
    src = patch(cand.read_text(), ROWS)
    out.write_text(src)
    print(out, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
