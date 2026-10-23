"""Vendor another problem's candidate into the layer's single-file submission.

    python dev/vendor.py <src.py> <prefix> > dev/vendored_<prefix>.py

Every top-level name the source defines (functions, classes, assignments) gets <prefix> in front,
everywhere it is used. Edits are made at the AST's source positions on the original text, so the
code keeps its formatting, comments and CUDA sources as written. The module docstring and the
`# okc:` header line are dropped. Fails if a function parameter or a comprehension/loop variable
shadows a renamed name (the rename would then be ambiguous)."""

import ast
import re
import sys

KEEP = {"torch", "triton", "tl", "math", "os", "types", "functools", "TypedDict", "F", "hashlib",
        "load_inline", "Optional"}


def defined_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
    return names - KEEP


def vendor(path, prefix):
    src = open(path).read()
    tree = ast.parse(src)
    names = defined_names(tree)
    lines = src.splitlines(keepends=True)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))

    def off(lineno, col):  # ast col offsets are in UTF-8 bytes; the sources here are ASCII
        return starts[lineno - 1] + col

    edits = []  # (start, end, new)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg in names:
            raise SystemExit(f"{path}:{node.lineno}: parameter {node.arg} shadows a top-level name")
        if isinstance(node, ast.Name) and node.id in names:
            s = off(node.lineno, node.col_offset)
            edits.append((s, s + len(node.id), prefix + node.id))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            line = lines[node.lineno - 1]
            m = re.search(r"\b(def|class)\s+(" + re.escape(node.name) + r")\b", line)
            assert m, (node.name, line)
            s = starts[node.lineno - 1] + m.start(2)
            edits.append((s, s + len(node.name), prefix + node.name))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            seg_s, seg_e = off(node.lineno, node.col_offset), off(node.end_lineno, node.end_col_offset)
            seg = src[seg_s:seg_e]
            for n in node.names:
                if n in names:
                    for m in re.finditer(r"\b" + re.escape(n) + r"\b", seg):
                        edits.append((seg_s + m.start(), seg_s + m.end(), prefix + n))
    # drop the module docstring
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        d = tree.body[0]
        edits.append((off(d.lineno, d.col_offset), off(d.end_lineno, d.end_col_offset), ""))
    out = src
    for s, e, new in sorted(set(edits), reverse=True):
        out = out[:s] + new + out[e:]
    out = "\n".join(ln for ln in out.splitlines() if not ln.startswith("# okc:"))
    ast.parse(out)  # still valid Python
    return out


if __name__ == "__main__":
    print(vendor(sys.argv[1], sys.argv[2]))
