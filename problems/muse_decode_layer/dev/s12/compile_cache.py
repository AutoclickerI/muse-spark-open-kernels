"""Reuse successful Hopper compiles only for identical source and compiler flags in this judge image.

The seed is S12 final's complete uncached normal local_eval result, pinned by its audit SHA.
Python static checks and runtime screens always run. New/changed CUDA strings use the original
local_eval compiler, at concurrency 2 because concurrent lane screens exhausted host memory.
"""
import ast
import hashlib
import inspect
import json
import tempfile
from pathlib import Path

H = Path(__file__).resolve().parent
BASE = H.parents[1] / 'cand_r3_best7_s12_final.py'
BASE_SHA = '240248b2c27c890bf5055a77e87d2fc9c4429b747a04d714e51d5f67e936bb3e'


def sources(path):
    tree = ast.parse(path.read_text())
    templated = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in ('format', 'replace', 'format_map') and isinstance(n.value, ast.Name):
            templated.add(n.value.id)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) and isinstance(n.left, ast.Name):
            templated.add(n.left.id)
    flags = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', getattr(n.func, 'id', '')) == 'load_inline':
            kw = {k.arg: k.value for k in n.keywords}
            src = kw.get('cuda_sources')
            names = [e.id for e in (src.elts if isinstance(src, (ast.List, ast.Tuple)) else [src]) if isinstance(e, ast.Name)]
            extra = []
            for key, prefix in (('extra_cuda_cflags', ''), ('extra_include_paths', '-I')):
                try:extra += [prefix + str(v) for v in ast.literal_eval(kw[key])]
                except (KeyError, ValueError):pass
            for name in names:flags[name] = extra
    found = {}
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            name, text = n.targets[0].id, n.value.value
            if '#include' in text and (name in flags or '__global__' in text or 'torch/extension.h' in text) and name not in templated:
                extra = [f for f in flags.get(name, []) if not f.startswith(('-gencode', '-arch', '--generate-code'))]
                found[name] = (text, extra, n)
    return found


def install(local_eval):
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
    root = H / 'final_screen/normal'
    assert json.loads((root / 'audit.json').read_text())['sha256'] == BASE_SHA
    seed = json.loads((root / 'normal_local.json').read_text())
    cached = {row['name']: row for row in seed['static'] if row['ok']}
    original = local_eval.hopper_compile
    # Exact compiler implementation; only reduce simultaneous compiler processes under memory pressure.
    code = inspect.getsource(original).replace('cf.ThreadPoolExecutor(8)', 'cf.ThreadPoolExecutor(2)')
    ns = dict(local_eval.__dict__)
    exec(compile(code, '<bounded-hopper-compile>', 'exec'), ns)
    compiler = ns['hopper_compile']
    base = sources(BASE)

    def check(path):
        current = sources(path)
        reuse = {name for name, (text, flags, _) in current.items()
                 if name in cached and name in base and (text, flags) == base[name][:2]}
        changed = {name for name in current if name not in reuse}
        rows = []
        if changed:
            lines = path.read_text().splitlines(keepends=True)
            for name in sorted(reuse, key=lambda k: current[k][2].lineno, reverse=True):
                n = current[name][2]
                lines[n.lineno-1:n.end_lineno] = [name + ' = ""\n']
            with tempfile.TemporaryDirectory(prefix='s13-static-') as tmp:
                reduced = Path(tmp) / path.name
                reduced.write_text(''.join(lines))
                rows = compiler(reduced)
            assert {r['name'] for r in rows} == changed, (changed, rows)
        byname = {r['name']: r for r in rows}
        print('HOPPER_EXACT_CACHE reused=' + ','.join(sorted(reuse)) + ' compiled=' + ','.join(sorted(changed)), flush=True)
        return [dict(cached[name], msg='identical CUDA and flags to verified ' + BASE_SHA[:8])
                if name in reuse else byname[name] for name in current]
    local_eval.hopper_compile = check
