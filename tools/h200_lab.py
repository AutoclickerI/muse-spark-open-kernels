#!/usr/bin/env python3
"""Run any H200 measurement inside the prepare() of a problem's best scored source (a lab run).

    tools/h200_lab.py send prof/lab/moe_skew.py --note "V9 vs grouped at real routing" [--host muse_moe_w4a16] [--priority 1]
    tools/h200_lab.py build prof/lab/moe_skew.py -o /tmp/x.py     # write the lab file only (to screen it)
    tools/h200_lab.py read <job | receipt.json>                    # print the lab's lines from a receipt

The payload is one self-contained file defining `lab(deadline, weights[, state]) -> list[str]`: `deadline` is a
time.monotonic() value (prepare's 300 s cap minus a margin, counted from prepare's start), `weights` is
what the host problem's prepare() receives (proj: qkv_gate, o, pooler, unpooler and the real router;
MoE: the packed NVFP4 experts). Inline its CUDA/Triton (load_inline under a unique name); no imports of
our modules, since the judge gets the one file. The optional third argument is the state returned by the
host's already-completed base prepare; it avoids rebuilding a long tuner when a lab measures that state.
The judge's image has torch, triton and SGLang's kernels.

The lab file is the host's best scored source (ledger) with the payload appended and prepare() wrapped:
the base prepare runs first and its state is returned unchanged, so the run still scores, and the board
keeps our best anyway. The judge calls prepare once per run (llms.txt) and keeps the last 16,000
characters of output, followed by up to ~6 KB of profiler lines, so the lab's lines are capped at
LIMIT characters between LAB_BEGIN and LAB_END. An exception in the payload becomes LAB_ERROR lines.

Hosts: proj and MoE have 30 runs a day that their own boards barely use (G3 counts only the decode
layer); decode, attention and prefill have 20, and attention has no prepare. Lab runs default to the
proj host at the priority of the problem the measurement serves (--priority).
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from screen_all import local_imports  # noqa: E402

LIMIT = 9000          # characters of lab output kept between the markers
MARGIN_S = 15         # left of prepare's 300 s after the payload's deadline
HOSTS = ("muse_bf16_proj", "muse_moe_w4a16", "muse_prefill_layer", "muse_decode_layer")

WRAPPER = '''
# ---- h200_lab wrapper: payload {name} sha256 {sha} on base {base} ----
_lab_base_prepare = prepare
_lab_ran = False


def prepare(weights):
    global _lab_ran
    import time as _t
    _start = _t.monotonic()
    state = _lab_base_prepare(weights)
    if not _lab_ran:
        _lab_ran = True
        import traceback as _tb
        _t0 = _t.monotonic()
        try:
            import inspect as _inspect
            _lab_params = len(_inspect.signature(lab).parameters)
            _lab_result = (lab(_start + {budget}, weights, state) if _lab_params >= 3
                           else lab(_start + {budget}, weights))
            _lines = [str(x) for x in (_lab_result or [])]
        except BaseException as _e:  # the run must still score; the error is the result
            _lines = [f"LAB_ERROR {{type(_e).__name__}}: {{_e}}"] + _tb.format_exc().splitlines()[-8:]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _out, _n = [], 0
        for _l in _lines:
            if _n + len(_l) + 1 > {limit}:
                _out.append(f"LAB_TRUNCATED {{len(_lines) - len(_out)}} lines")
                break
            _out.append(_l)
            _n += len(_l) + 1
        print("LAB_BEGIN {name} {sha}", flush=True)
        for _l in _out:
            print(_l, flush=True)
        print(f"LAB_END lab_s={{_t.monotonic() - _t0:.1f}} prepare_s={{_t.monotonic() - _start:.1f}}", flush=True)
    return state
'''


def best_source(problem: str) -> dict:
    rows = [json.loads(l) for l in (REPO / "runs/ledger.jsonl").open()]
    done = [r for r in rows if r["problem"] == problem and r.get("status") == "done" and r.get("speedup")
            and r["mode"] == "submit" and "h200_lab" not in r.get("note", "")]
    return max(done, key=lambda r: r["speedup"])


def build(payload: Path, host: str) -> tuple[str, str]:
    if bad := local_imports(payload):
        sys.exit(f"{payload}: imports modules the judge won't have: {sorted(bad)}")
    body = payload.read_text()
    if not re.search(r"^def lab\(", body, re.M):
        sys.exit(f"{payload}: must define lab(deadline, weights) -> list[str]")
    base = best_source(host)
    src = (REPO / base["source"]).read_text()
    if not re.search(r"^def prepare\(", src, re.M):
        sys.exit(f"{host} best source {base['source']} has no prepare(); pick another --host")
    sha = hashlib.sha256(body.encode()).hexdigest()[:16]
    out = (src.rstrip() + f"\n\n# ---- h200_lab payload {payload.name} ----\n" + body.rstrip() + "\n"
           + WRAPPER.format(name=payload.stem, sha=sha, base=base["job"], budget=300 - MARGIN_S, limit=LIMIT))
    return out, base["job"]


def read(ref: str):
    p = Path(ref)
    if not p.exists():
        hits = sorted(REPO.glob(f"runs/*/*_{ref}.json"))
        if not hits:
            sys.exit(f"no receipt for {ref} (still running? `tools/okc_run.py collect {ref}`)")
        p = hits[-1]
    tail = json.loads(p.read_text())["result"].get("log_tail", "")
    m = re.search(r"^LAB_BEGIN.*?(?:^LAB_END.*?$|\Z)", tail, re.M | re.S)
    print(m.group(0) if m else f"no LAB_BEGIN in {p} (log_tail {len(tail)} chars):\n{tail[-2000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("send", "build"):
        s = sub.add_parser(name)
        s.add_argument("payload", type=Path)
        s.add_argument("--host", default="muse_bf16_proj", choices=HOSTS)
        if name == "send":
            s.add_argument("--note", required=True)
            s.add_argument("--priority", type=float, default=1.0)
        else:
            s.add_argument("-o", type=Path, required=True)
    r = sub.add_parser("read")
    r.add_argument("ref", help="job id or receipt .json")
    args = ap.parse_args()

    if args.cmd == "read":
        return read(args.ref)
    src, base = build(args.payload, args.host)
    if args.cmd == "build":
        args.o.write_text(src)
        return print(f"{args.o}: {len(src)} bytes, base {base}")
    out = REPO / "problems" / args.host / "lab" / f"lab_{args.payload.stem}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    subprocess.run([sys.executable, str(REPO / "tools/okc_run.py"), "send", str(out.relative_to(REPO)),
                    "--priority", str(args.priority), "--note", f"h200_lab {args.payload}: {args.note}"],
                   cwd=REPO, check=False)


if __name__ == "__main__":
    main()
