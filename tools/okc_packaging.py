"""Single-file judge packaging checks shared by local preflight and lab runners.

Keep this helper separate from CLI modules: integration lanes also define screen_all.
"""
import ast
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]

# The judge gets the one file plus the problem's official files (okc init), so an import of any other module
# of ours passes here and fails there.
OFFICIAL = {"reference", "baseline", "checks", "nvfp4"}


def local_imports(f: Path) -> set:
    ours = ({p.stem for p in REPO.glob("problems/**/*.py")} | {p.stem for p in REPO.glob("tools/*.py")}
            | {p.stem for p in REPO.glob("sim/*.py")} | {"sim", "prof", "tools"}) - OFFICIAL
    found = set()
    for n in ast.walk(ast.parse(f.read_text())):
        if isinstance(n, ast.Import):
            found |= {a.name for a in n.names if a.name.split(".")[0] in ours}
        elif isinstance(n, ast.ImportFrom) and (n.level or (n.module or "").split(".")[0] in ours):
            found.add("." * n.level + (n.module or ""))
    return found
