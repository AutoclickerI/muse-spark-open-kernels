"""Extract the current candidate's CUDA/C++ source for native compilation and SASS inspection.

Run in the judge image. The complete sm_90a compilation gate is tools/local_eval.py;
this command materializes the exact embedded source without importing the candidate.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path,
                        default=ROOT / "problems/muse_decode_layer/cand_r3_best7_s14_final.py")
    parser.add_argument("--out", type=Path, default=ROOT / "prof/sass")
    args = parser.parse_args()
    source = args.candidate.read_bytes()
    manifest = {"source": str(args.candidate.resolve().relative_to(ROOT)),
                "sha256": hashlib.sha256(source).hexdigest(), "sources": {}}
    args.out.mkdir(parents=True, exist_ok=True)
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        name, text = node.targets[0].id, node.value.value
        if not (name.endswith(("CUDA", "CUDA_SRC", "CPP", "CPP_SRC"))
                and ("#include" in text or "__global__" in text)):
            continue
        suffix = ".cpp" if name.endswith(("CPP", "CPP_SRC")) else ".cu"
        path = args.out / (name + suffix)
        path.write_text(text)
        manifest["sources"][name] = {"file": path.name,
                                     "sha256": hashlib.sha256(text.encode()).hexdigest()}
    assert manifest["sources"], "candidate contains no embedded CUDA/C++ source"
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Extracted {len(manifest['sources'])} exact source strings from {manifest['sha256'][:8]}")


if __name__ == "__main__":
    main()
