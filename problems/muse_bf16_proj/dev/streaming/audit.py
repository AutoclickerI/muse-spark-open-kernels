"""Compile the actual lab's CUDA for H200 and check its load/MMA/spill evidence."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("payload", nargs="?", type=Path, default=HERE.parent / "streaming_gemv_lab.py")
parser.add_argument("--output", type=Path, default=HERE / "compile_sm90a.json")
args = parser.parse_args()
tree = ast.parse(args.payload.read_text())
src = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "SP_CUDA" for t in n.targets))
scratch = REPO / "runs/screen/streaming"
scratch.mkdir(parents=True, exist_ok=True)
tag = hashlib.sha256(src.encode()).hexdigest()[:16]
cu, obj = scratch / f"{tag}.cu", scratch / f"{tag}.o"
cu.write_text(src)
command = [str(REPO / "tools/img"), "/usr/local/cuda/bin/nvcc", "-arch=sm_90a",
           "-std=c++17", "-O3", "-lineinfo", "--ptxas-options=-v", "-c", str(cu), "-o", str(obj)]
compiled = subprocess.run(command, capture_output=True, text=True, check=True)
sass = subprocess.run([str(REPO / "tools/img"), "/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(obj)],
                      capture_output=True, text=True, check=True).stdout
functions = []
for block in re.split(r"ptxas info    : Compiling entry function ", compiled.stderr)[1:]:
    name = re.search(r"'([^']+)'", block)[1]
    regs = int(re.search(r"Used (\d+) registers", block)[1])
    spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", block)
    stores, loads = map(int, spills.groups())
    if stores or loads:
        raise RuntimeError(f"spilling in {name}: {stores}/{loads}")
    functions.append(dict(name=name, registers=regs, spill_stores=stores, spill_loads=loads))
if len(functions) != 20:
    raise RuntimeError(f"expected 19 projection kernels + read control, got {len(functions)}")
rings = {}
for u, rows, warps in ((4,1,4), (8,1,4), (16,1,4), (8,1,8), (4,2,4), (8,2,4)):
    prefix = f"_ZN11stream_proj6scalarILi{u}ELi{warps}ELi{rows}"
    body = next(b for b in sass.split("Function : ") if b.startswith(prefix))
    before = body.split("FFMA")[0]
    loads = len(re.findall(r"LDG\.E\.128\.STRONG\.GPU", before))
    if loads < u * rows:
        raise RuntimeError(f"register ring collapsed: U={u} rows={rows} loads={loads}")
    rings[f"u{u}_r{rows}_w{warps}"] = loads
counts = {op: len(re.findall(op, sass)) for op in
          (r"LDG\.E\.128", r"LDGSTS\.E\.BYPASS\.128", r"HMMA\.16816\.F32\.BF16")}
if not all(counts.values()):
    raise RuntimeError(f"missing vector load/async copy/BF16 MMA: {counts}")
out = dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(), arch="sm_90a",
           kernels=functions, sass_counts=counts, weight_loads_before_first_fma=rings)
args.output.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(dict(kernels=len(functions), max_registers=max(f["registers"] for f in functions),
                      no_spills=True, rings=rings)))
