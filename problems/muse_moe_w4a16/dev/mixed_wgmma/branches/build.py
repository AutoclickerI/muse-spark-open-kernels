"""Assemble the M256 and M128 branch/remainder lab payloads from frozen S6 / H_H sources."""
import ast
import hashlib
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parent
R = H.parents[4]
sys.path[:0] = [str(P), str(R / 'tools'), str(H)]
from build_prefill_lab import subset  # noqa: E402
from okc_run import rejected_words  # noqa: E402
import sources  # noqa: E402

HH_CELL = P / 'm128/phase_mix/hh_cell.py'
HH_SHA = '365b623992367a1d45d3dae06a2096ddd0e6879deffab2e3924678a47997b621'
IMPORTS = ('import functools\nimport hashlib\nimport statistics\nimport time\nimport torch\nimport triton\n'
           'import triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\n')


def strip_imports(code):
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Try))]
    return ast.unparse(ast.fix_missing_locations(tree))


def helpers():
    return subset((R / 'prof/lab/moe_w2fix_m256.py').read_text(), ('dc_weights', 'dc_fixtures', 'dc_oracles', 'MWGraph'))


def write(name, body, meta):
    bad = rejected_words(body)
    assert not bad, bad
    compile(body, name, 'exec')
    out = R / 'prof/lab' / f'{name}.py'
    out.write_text(body)
    meta = dict(payload=str(out.relative_to(R)), sha256=hashlib.sha256(body.encode()).hexdigest(), **meta)
    print(meta['payload'], meta['sha256'], len(body))
    return meta


def dupes(body):
    seen = {}
    for n in ast.parse(body).body:
        names = [n.name] if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else [getattr(x, 'id', None) for x in getattr(n, 'targets', [])]
        for k in names:
            if k:
                seen.setdefault(k, set()).add(ast.dump(n))
    return sorted(k for k, v in seen.items() if len(v) > 1)


def main():
    src = sources.sources()
    text = sources.s6_text()
    hh = HH_CELL.read_text()
    assert hashlib.sha256(hh.encode()).hexdigest() == HH_SHA
    s7 = sources.s6_value((R / 'problems/muse_decode_layer/cand_r3_best7_s7.py').read_text(), 'S5_HH_CUDA')
    cell = sources.s6_value(hh, 'M128_HH_PM_CUDA')
    assert 'unsigned long long sf' not in cell and '__byte_perm(raw,0u,0x4040u)*16u+0x70007000u' in cell and cell.count('void dq(') == 1
    assert cell[cell.index('void dq('):cell.index('void dq(') + 600] == s7[s7.index('void dq('):s7.index('void dq(') + 600], 'hh_cell dq must be S7\'s'
    u16 = sources.s6_value(text, 'S5_HH_CUDA')
    engine = strip_imports(subset(text, ('wi_prepare', 'wi_plan', 'wi_call')))
    common = (f'BR_HH_CUDA = {src["cu_hh"]!r}\nBR_HH_CPP = {src["cpp_hh"]!r}\nBR_S6_CUDA = {src["cu_s6"]!r}\nBR_S6_CPP = {src["cpp_s6"]!r}\n'
              f'BR_U16_CUDA = {u16!r}\n')
    manifest = dict(S6_sha256=sources.S6_SHA, HH_cell_sha256=HH_SHA, cu_hh_sha256=hashlib.sha256(src['cu_hh'].encode()).hexdigest(),
                    cu_s6_sha256=hashlib.sha256(src['cu_s6'].encode()).hexdigest(), u16_sha256=hashlib.sha256(u16.encode()).hexdigest())
    for name, kind in (('moe_br_m', 'branch'), ('moe_nf_tune_m', 'tune')):
        for m in (128, 256):
            body = (helpers() + '\n' + IMPORTS + strip_imports(hh) + '\n' + (engine + '\n' if m == 256 and kind == 'branch' else '')
                    + common + f'BR_M = {m}\nBR_SET = {kind!r}\n' + (H / 'plan.py.inc').read_text() + '\n' + (H / 'driver.py.inc').read_text())
            assert not dupes(body), dupes(body)
            manifest[f'{name}{m}'] = write(f'{name}{m}', body, {})
    (H / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
