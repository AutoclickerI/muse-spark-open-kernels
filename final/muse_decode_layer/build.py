#!/usr/bin/env python3
"""The judged S22 decode layer as readable files, and the proof that they are the judged bytes.

The scored file, problems/muse_decode_layer/cand_r3_best7_s22p1_final.py (job 210d40e0c9cb4b71,
1.8892x), is a short wrapper around an LZMA + base85 blob. Unpacked, the blob is one 1.7 MB
Python file whose CUDA sources and vendored Python modules mostly sit in string literals on single
lines; one vendored module is packed again, as base64 of zlib.

`split` writes that source as layer.py with every large single-line string literal (at least
MIN_CHARS characters, written as Python's repr) moved to a file under src/. In layer.py, and inside
the Python files under src/ (modules exec()ed from strings), the literal becomes _embedded("<path>"),
and layer.py's _embedded() returns the original string. A base64-of-zlib literal becomes
_embedded("<path>", zlib=True): src/<path>.b64 keeps the literal exactly and src/<path> its decoded,
readable module.

`check` joins the files back and verifies that the result is the judged source byte for byte, that
layer.py's own _embedded() rebuilds the same strings, that every .b64 literal decodes to its readable
module, and that the scored file's blob unpacks to the joined source. It needs no particular zlib or
liblzma build: nothing is recompressed.

    python3 final/muse_decode_layer/build.py check
    python3 final/muse_decode_layer/build.py split    # regenerate layer.py and src/ from the scored file
"""

import ast
import base64
import hashlib
import lzma
import re
import shutil
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
LAYER = HERE / "layer.py"
SHIPPED = HERE.parents[1] / "problems/muse_decode_layer/cand_r3_best7_s22p1_final.py"
SHIPPED_SHA256 = "2287496067a3da750d43afd1943c12e69fb1d5337ce1c5e528d4b6c64649a6fd"
SOURCE_SHA256 = "fd5e0ca2ef4bbc1128b04b68aa6802c9543b61808e7663ca5ad10ec4d957110b"
MIN_CHARS = 2000
PLACEHOLDER = re.compile(r'_embedded\("([^"]+)"(, zlib=True)?\)')
LOADER = '''
# ---- Readable split (final/muse_decode_layer/build.py): large source strings live in src/.
# _embedded() returns each original string; `build.py check` proves the join is byte-exact.
def _embedded(path, zlib=False):
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parent / "src"
    if zlib:  # the exact base64-of-zlib literal; src/<path> holds its decoded module
        return (root / (path + ".b64")).read_bytes().decode()
    text = (root / path).read_bytes().decode()
    if path.endswith(".py"):
        text = re.sub(r'_embedded\\("([^"]+)"(, zlib=True)?\\)',
                      lambda m: repr(_embedded(m.group(1), bool(m.group(2)))), text)
    return text
# ---- end of readable split
'''


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def judged_source():
    """The source the judge ran, unpacked from the scored file."""
    shipped = SHIPPED.read_bytes()
    assert sha256(shipped) == SHIPPED_SHA256, f"{SHIPPED} is not the scored file"
    match = re.search(rb"b85decode\('([^']+)'\)", shipped)
    assert match, f"{SHIPPED} has no packed source"
    source = lzma.decompress(base64.b85decode(match.group(1))).decode()
    assert sha256(source) == SOURCE_SHA256, "the scored file does not unpack to the judged source"
    return source


def zlib_decode(literal):
    """The text packed in a base64-of-zlib literal, or None."""
    try:
        return zlib.decompress(base64.b64decode(literal, validate=True)).decode()
    except (ValueError, zlib.error):
        return None


def kind(text):
    try:
        ast.parse(text)
        return ".py"
    except (SyntaxError, ValueError):
        pass
    if any(mark in text for mark in ("#include", "__global__", "__device__", "#define")):
        return ".cu"
    return ".txt"


def literal_names(tree):
    """A readable file name for string constants: the assigned name, dict key or compile() filename."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if isinstance(node.value, ast.Constant):
                names[id(node.value)] = target
            elif isinstance(node.value, ast.Dict):
                for key, value in zip(node.value.keys, node.value.values):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        names[id(value)] = f"{target}/{key.value}"
        elif (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "compile" and len(node.args) > 1
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            for inner in ast.walk(node.args[0]):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    names.setdefault(id(inner), node.args[1].value.strip("<>"))
    return {key: "/".join(re.sub(r"[^A-Za-z0-9_-]", "_", part) for part in name.split("/"))
            for key, name in names.items()}


def split_text(text, folder, files):
    """Move text's large repr-written string literals to files[folder + name]; return the new text."""
    assert not PLACEHOLDER.search(text)
    tree = ast.parse(text)
    data = text.encode()
    starts = [0]
    for line in data.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    names = literal_names(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= MIN_CHARS:
            a = starts[node.lineno - 1] + node.col_offset
            b = starts[node.end_lineno - 1] + node.end_col_offset
            if data[a:b] == repr(node.value).encode():
                found.append((a, b, node))
    out, pos = [], 0
    for a, b, node in sorted(found, key=lambda f: f[0]):
        packed = zlib_decode(node.value)
        body = node.value if packed is None else packed
        ext = kind(body)
        stem = folder + names.get(id(node), f"L{node.lineno}")
        path = stem + ext
        if path in files:
            path = f"{stem}_L{node.lineno}{ext}"
        assert path not in files, path
        files[path] = None
        files[path] = split_text(body, stem + "/", files) if ext == ".py" else body
        if packed is None:
            call = f'_embedded("{path}")'
        else:
            files[path + ".b64"] = node.value
            call = f'_embedded("{path}", zlib=True)'
        out += [data[pos:a], call.encode()]
        pos = b
    out.append(data[pos:])
    return b"".join(out).decode()


def load(path, packed, used):
    used.add(path)
    text = (SRC / path).read_bytes().decode()
    if path.endswith(".py"):
        text = join_text(text, used)
    if not packed:
        return text
    used.add(path + ".b64")
    literal = (SRC / (path + ".b64")).read_bytes().decode()
    assert zlib_decode(literal) == text, f"src/{path}.b64 does not decode to src/{path}"
    return literal


def join_text(text, used):
    return PLACEHOLDER.sub(lambda m: repr(load(m.group(1), bool(m.group(2)), used)), text)


def check():
    layer = LAYER.read_bytes().decode()
    assert layer.count(LOADER) == 1, "layer.py lost its _embedded loader"
    used = set()
    joined = join_text(layer.replace(LOADER, "", 1), used)
    assert joined == judged_source(), "joined source differs from the judged source"
    on_disk = {p.relative_to(SRC).as_posix() for p in SRC.rglob("*") if p.is_file()}
    assert on_disk == used, f"unreferenced files under src/: {sorted(on_disk - used)}"
    runtime: dict = {"__file__": str(LAYER)}
    exec(LOADER, runtime)
    rebuilt = PLACEHOLDER.sub(lambda m: repr(runtime["_embedded"](m.group(1), bool(m.group(2)))),
                              layer.replace(LOADER, "", 1))
    assert rebuilt == joined, "layer.py's _embedded() does not rebuild the judged strings"
    print(f"ok: layer.py + {len(used)} files under src/ join to the judged source "
          f"(sha256 {SOURCE_SHA256[:12]}, unpacked from the scored file), and layer.py's _embedded() "
          "rebuilds the same strings")


def split():
    source = judged_source()
    assert "_embedded" not in source
    files = {}
    text = split_text(source, "", files)
    docstring_end = ast.parse(source).body[0].end_lineno
    lines = text.splitlines(keepends=True)
    text = "".join(lines[:docstring_end]) + LOADER + "".join(lines[docstring_end:])
    shutil.rmtree(SRC, ignore_errors=True)
    for path, body in files.items():
        target = (SRC / path).resolve()
        assert target.is_relative_to(SRC), path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body.encode())
    LAYER.write_bytes(text.encode())
    print(f"wrote layer.py ({text.count(chr(10))} lines) and {len(files)} files under src/")
    check()


if __name__ == "__main__":
    {"split": split, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
