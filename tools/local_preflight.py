"""Static checks only: never imports the candidate, Torch, reference or GPU code."""
import ast
import base64
import hashlib
import lzma
from pathlib import Path
import tempfile

def undefined_names(file: Path, known=()) -> list[dict]:
    """Names a function uses that nothing defines (ruff F821-F823): a branch that never runs here
    still raises NameError on the judge (cand_r3_merge 43a1f3a3 lost every XQA variant to
    ATTN_X_CUDA). Ruff lives in the image's home (~/bin/ruff, a copy of the host's)."""
    import os
    import shutil
    import subprocess
    ruff = shutil.which("ruff") or os.path.expanduser("~/bin/ruff")
    p = subprocess.run([ruff, "check", "--no-cache", "--isolated", "--select", "F821,F822,F823",
                        "--config", "builtins="+repr(list(known)), "--output-format", "concise", str(file)], capture_output=True, text=True)
    found = [l.split(": ", 1)[-1] for l in p.stdout.splitlines() if ": F82" in l]
    return [{"name": "undefined_names", "ok": p.returncode == 0, "msg": " | ".join(found[:8])[:400]}]


def call_signatures(file: Path) -> list[dict]:
    """Calls to the file's own module-level functions that pass an argument they don't take: a
    Hopper-only branch raises TypeError only on the judge (cand_attn_verify_wgmma 05fe4997:
    _next_input(..., ctx=2048)). Names bound more than once at module level are skipped."""
    tree = ast.parse(file.read_text())
    bound: dict[str, int] = {}
    defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        for target in (node.targets if isinstance(node, ast.Assign) else []):
            if isinstance(target, ast.Name):
                bound[target.id] = bound.get(target.id, 0) + 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound[node.name] = bound.get(node.name, 0) + 1
            defs[node.name] = node
    found = []
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            continue
        fn = defs.get(call.func.id)
        if fn is None or bound[fn.name] > 1 or any(isinstance(d, ast.Name) and d.id != "staticmethod"
                                                    for d in fn.decorator_list):
            continue
        a = fn.args
        names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        positional = len(a.posonlyargs + a.args)
        if any(isinstance(x, ast.Starred) for x in call.args) or any(k.arg is None for k in call.keywords):
            continue
        bad = [str(k.arg) for k in call.keywords if a.kwarg is None and k.arg not in names]
        if a.vararg is None and len(call.args) > positional:
            bad.append(f"{len(call.args)} positional > {positional}")
        if bad:
            found.append(f"line {call.lineno}: {fn.name}() {', '.join(bad)}")
    return [{"name": "call_signatures", "ok": not found, "msg": " | ".join(found[:8])[:400]}]



def source_units(source):
    return {n.name if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else n.targets[0].id: ast.dump(n)
            for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            or (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name))}


def expanded_source(source):
    # Inspect our literal, checksummed pack format without executing any user code.
    tree = ast.parse(source)
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == '_source' for t in n.targets):
            call = n.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'decode'):
                continue
            try:
                packed = call.func.value.args[0].args[0].value
                body = lzma.decompress(base64.b85decode(packed)).decode()
            except (AttributeError, IndexError, TypeError, ValueError, lzma.LZMAError) as exc:
                raise ValueError('invalid packed source') from exc
            digest = hashlib.sha256(body.encode()).hexdigest()
            if digest not in source:
                raise ValueError('packed source checksum mismatch or missing')
            lines = source.splitlines(True)
            # Replace the exec site for analysis, retaining wrapper code around it as well.
            for call in tree.body:
                if (isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
                        and isinstance(call.value.func, ast.Name) and call.value.func.id == 'exec'
                        and any(isinstance(arg, ast.Name) and arg.id == '_source'
                                for arg in ast.walk(call.value))):
                    return ''.join(lines[:call.lineno-1]) + body + '\n' + ''.join(lines[call.end_lineno:])
            raise ValueError('packed source has no recognized exec site')
    return source


def state_keys(source):
    tree = ast.parse(source)
    prepares = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare']
    kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn']
    if not prepares or not kernels:
        return [{'name': 'state_keys', 'ok': True, 'msg': 'no literal prepare state contract'}]
    returns = [n.value for n in ast.walk(prepares[-1]) if isinstance(n, ast.Return)]
    # Only closed literal dictionaries establish a statically complete key set. Delegated/dynamic state
    # is reported as unresolved, never guessed from unrelated dictionaries in the file.
    if not returns or not all(isinstance(r, ast.Dict) and all(isinstance(k, ast.Constant) for k in r.keys) for r in returns):
        return [{'name': 'state_keys', 'ok': True, 'msg': 'dynamic state contract: requires changed-unit/H200 checks'}]
    keys = set.intersection(*[{k.value for k in r.keys} for r in returns])
    args = kernels[-1].args.posonlyargs + kernels[-1].args.args
    state = args[1].arg if len(args) > 1 else None
    used = {n.slice.value for n in ast.walk(kernels[-1]) if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name) and n.value.id == state and isinstance(n.slice, ast.Constant)}
    missing = used - keys
    return [{'name': 'state_keys', 'ok': not missing, 'msg': 'missing prepare keys: '+repr(sorted(missing)) if missing else ''}]


def preflight(file, *, payload=False):
    from screen_all import local_imports
    from okc_run import rejected_words
    source = file.read_text()
    checks = []
    try:
        compile(source, str(file), 'exec')
        expanded = expanded_source(source)
        compile(expanded, '<expanded>', 'exec')
    except (SyntaxError, ValueError) as exc:
        return [{'name': 'syntax/packaging', 'ok': False, 'msg': str(exc)}]
    import re
    checks.append(dict(name='packaging', ok=len(source.encode()) < 1_000_000 and
                       (payload or bool(re.match(r'#\s*okc:\s*problem=\w+', source))), msg='single-file problem header and size'))
    with tempfile.TemporaryDirectory(prefix='okc-preflight-') as tmp:
        path = Path(tmp)/'expanded.py'
        path.write_text(expanded)
        bad = local_imports(path)
        checks.append(dict(name='local_imports', ok=not bad, msg=repr(sorted(bad)) if bad else ''))
        hits = rejected_words(source) + (rejected_words(expanded) if expanded != source else [])
        checks.append(dict(name='judge_packaging', ok=not hits, msg=' | '.join(hits[:5])))
        checks += undefined_names(path, known=('lab_cold_l2','kernel_fn','prepare') if payload else ()) + call_signatures(path) + state_keys(expanded)
    return checks
