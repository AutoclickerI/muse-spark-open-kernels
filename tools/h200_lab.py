#!/usr/bin/env python3
"""Run H200 measurements in prepare() or at import of a problem's best scored source (a lab run).

    tools/h200_lab.py send prof/lab/moe_skew.py [more.py ...] --note "V9 vs grouped at real routing" [--host muse_moe_w4a16] [--priority 1]
    tools/h200_lab.py build prof/lab/moe_skew.py [more.py ...] -o /tmp/x.py   # write the lab file only (to screen it)
    tools/h200_lab.py read <job | receipt.json> [name]                        # print the lab sections from a receipt

The payload is one self-contained file defining `lab(deadline, weights[, state]) -> list[str]`: `deadline` is a
time.monotonic() deadline (prepare's 300 s cap minus a margin, or --import-budget seconds for hosts
without prepare). Payloads must honor it; it does not interrupt running code. `weights` is
what the host problem's prepare() receives (proj: qkv_gate, o, pooler, unpooler and the real router;
MoE: the packed NVFP4 experts). Inline its CUDA/Triton (load_inline under a unique name); no imports of
our modules, since the judge gets the one file. The optional third argument is the state returned by the
host's already-completed base prepare; it avoids rebuilding a long tuner when a lab measures that state.
Without prepare(), payloads run once after the host's definitions at module import, with weights=None and
state=None; payloads needing weights must synthesize them. The import budget starts at the wrapper,
after any import work already present in the best source. Neither kernel nor kernel_fn is changed.
The judge's image has torch, triton and SGLang's kernels.

The lab file is the host's best scored source (ledger) with prepare() wrapped: the base prepare runs
first and its state is returned unchanged, so the run still scores, and the board keeps our best anyway.
Each payload's source is embedded as a string and executed in its own namespace (a copy of the host
module's globals, so host functions stay reachable and payload names can't collide), one after another,
each with an equal share of the time left. Payloads make L2 cold with `lab_cold_l2()` (a read, as the judge
does), never with a zero_() write. Several payloads for one host go in one judge run: our runs
start one at a time, so batching labs is the cheapest way to more H200 answers per hour. `send` also folds
in a lab still waiting for a judge slot on the same host (up to MERGE_MAX payloads per run; a payload of
the same name is replaced by the new one), so lanes that send separately still share runs. The judge calls
prepare once per run (llms.txt) and keeps the last 16,000 characters of output, followed by up to ~6 KB
of profiler lines, so all lab lines together are capped at LIMIT characters, split evenly; each payload
prints between its own LAB_BEGIN and LAB_END; over its share, the middle lines go (the first two and the last are kept). An exception in a payload becomes its LAB_ERROR lines; a sticky
CUDA error (lab b63389f0) still fails the run after printing them and kills the payloads after it, so put a
risky payload last.

Hosts: proj, MoE, prefill, decode and attention, sharing each host's board quota. Prefill is off the
overall board. Attention has no judge-called prepare(), so uses import-time labs (--import-budget,
default 45 s). Lab runs default to proj at the priority of the measurement's problem (--priority).
"""

import argparse
import ast
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from screen_all import local_imports  # noqa: E402

LIMIT = 9000          # characters of lab output kept between the markers
MARGIN_S = 15         # left of prepare's 300 s after the payload's deadline
HOSTS = ("muse_bf16_proj", "muse_moe_w4a16", "muse_prefill_layer", "muse_decode_layer", "muse_attention_decode")
IMPORT_BUDGET_S = 45  # H200 probe cd9e07cf650e4839 completed 45 s; the parent's maximum is unknown
MERGE_MAX = 2        # payloads per run: base prepares take ~130 s on the H200, leaving each ~80 s

WRAPPER = '''
# ---- h200_lab wrapper: payloads {names} on base {base} ----
_LAB_PAYLOADS = {payloads!r}
_lab_ran = False
_lab_flush_buf = None


def lab_cold_l2():
    # The judge's cold-L2 step (TASK.md): read a buffer larger than L2 before each timed replay. Never flush with
    # zero_()/fill_(): that leaves ~50 MB of dirty lines for the timed call to write back (+5-16 us at M=1-32).
    global _lab_flush_buf
    if _lab_flush_buf is None:
        _lab_flush_buf = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    _lab_flush_buf.view(torch.int64).sum()  # int64 view: the uint8 sum upcasts and runs ~4x slower


def _lab_run_one(name, src, deadline, weights, state):
    import inspect as _inspect, linecache as _linecache
    fname = f"<h200_lab {{name}}>"  # in linecache, so inspect.getsource (triton.jit) finds the source
    _linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns = dict(globals())
    exec(compile(src, fname, "exec"), ns)
    fn = ns["lab"]
    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)


def _lab_run(weights, state, _start, _end, _phase):
    global _lab_ran
    import time as _t
    if not _lab_ran:
        _lab_ran = True
        import traceback as _tb
        for _i, (_name, _sha, _src) in enumerate(_LAB_PAYLOADS):
            _t0 = _t.monotonic()
            _deadline = _t0 + max(0.0, _end - _t0) / (len(_LAB_PAYLOADS) - _i)
            _lines = []
            try:
                if _t0 >= _end:
                    raise TimeoutError("lab budget exhausted before this payload")
                _lines = [str(x) for x in (_lab_run_one(_name, _src, _deadline, weights, state) or [])]
                torch.cuda.synchronize()  # inside the try: a sticky CUDA error still prints this payload's lines
                torch.cuda.empty_cache()
            except BaseException as _e:  # the error is the result; a sticky CUDA error still fails the run after
                _lines += [f"LAB_ERROR {{type(_e).__name__}}: {{_e}}"] + _tb.format_exc().splitlines()[-8:]
                if weights is None:
                    _lines += ["LAB_ERROR host supplies weights=None; payloads needing weights must synthesize them "
                               "(state is also None on import-time hosts)"]
            # Over the cap, keep the first two lines (META) and the last lines (summaries, errors) and drop the middle.
            _cap = {limit} // len(_LAB_PAYLOADS)
            _head, _tail, _n = _lines[:2], [], sum(len(_l) + 1 for _l in _lines[:2]) + 40
            if sum(len(_l) + 1 for _l in _lines) > _cap:
                for _l in reversed(_lines[2:]):
                    if _n + len(_l) + 1 > _cap:
                        break
                    _tail.insert(0, _l)
                    _n += len(_l) + 1
                _out = _head + [f"LAB_TRUNCATED {{len(_lines) - 2 - len(_tail)}} middle lines"] + _tail
            else:
                _out = _lines
            print(f"LAB_BEGIN {{_name}} {{_sha}}", flush=True)
            for _l in _out:
                print(_l, flush=True)
            print(f"LAB_END lab_s={{_t.monotonic() - _t0:.1f}} {{_phase}}_s={{_t.monotonic() - _start:.1f}}", flush=True)
'''

PREPARE_WRAPPER = '''
_lab_base_prepare = prepare


def prepare(weights):
    import time as _t
    _start = _t.monotonic()
    state = _lab_base_prepare(weights)
    _lab_run(weights, state, _start, _start + {budget}, "prepare")
    return state
'''

IMPORT_WRAPPER = '''
import time as _lab_time
_lab_import_start = _lab_time.monotonic()
_lab_run(None, None, _lab_import_start, _lab_import_start + {budget}, "import")
'''


def best_source(problem: str) -> dict:
    rows = [json.loads(l) for l in (REPO / "runs/ledger.jsonl").open()]
    def withdrawn(row):
        if row.get("withdrawn"):
            return True
        receipt = REPO / Path(row["source"]).with_suffix(".json")
        return receipt.exists() and json.loads(receipt.read_text()).get("withdrawn", False)
    done = [r for r in rows if r["problem"] == problem and r.get("status") == "done" and r.get("speedup")
            and r["mode"] == "submit" and "h200_lab" not in r.get("note", "") and not withdrawn(r)]
    return max(done, key=lambda r: r["speedup"])


IMPORT_CHECK = '''
import ast, importlib, importlib.util, sys
CATCH = {"ImportError", "ModuleNotFoundError"}  # a catch-all is error reporting, not an optional import
def guarded(t):
    """A try whose handlers catch import failures: its imports are optional by design."""
    return any(h.type is not None and any(isinstance(n, ast.Name) and n.id in CATCH for n in ast.walk(h.type))
               for h in t.handlers)
def imports(node, optional=False):
    if isinstance(node, ast.Try) and guarded(node):
        for n in node.body:
            yield from imports(n, True)
        for n in node.handlers + node.orelse + node.finalbody:
            yield from imports(n, optional)
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)) and not optional:
        yield node
    for n in ast.iter_child_nodes(node):
        yield from imports(n, optional)
bad = []
for node in imports(ast.parse(open(sys.argv[1]).read())):
    if isinstance(node, ast.ImportFrom) and node.module and not node.level:
        try:
            mod = importlib.import_module(node.module)
            bad += [f"{node.module}.{a.name}" for a in node.names if a.name != "*" and not hasattr(mod, a.name)
                    and importlib.util.find_spec(f"{node.module}.{a.name}") is None]
        except Exception as e:
            bad.append(f"{node.module}: {type(e).__name__}: {e}")
    elif isinstance(node, ast.Import):
        for a in node.names:
            try:
                importlib.import_module(a.name)
            except Exception as e:
                bad.append(f"{a.name}: {type(e).__name__}: {e}")
print("BAD " + " | ".join(bad) if bad else "OK")
'''


def check_imports(payload: Path) -> list[str]:
    """Every import in the payload, nested ones included, resolved in the judge's image (tools/img),
    except those inside a try that catches import failures (optional by design):
    humming_nvfp4_probe fbf64bd3 spent a judge slot on a wrong module path."""
    p = subprocess.run([str(REPO / "tools/img"), "python", "-c", IMPORT_CHECK, str(payload.resolve())],
                       cwd=REPO, capture_output=True, text=True, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith(("OK", "BAD "))), None)
    if line is None:
        return [f"import check did not run: {p.stderr[-400:]}"]
    return [] if line == "OK" else line[4:].split(" | ")


def embed(payload: Path) -> tuple[str, str, str]:
    if bad := local_imports(payload):
        sys.exit(f"{payload}: imports modules the judge won't have: {sorted(bad)}")
    if bad := check_imports(payload):
        sys.exit(f"{payload}: imports that fail in the judge's image: {bad}")
    body = payload.read_text()
    if not re.search(r"^def lab\(", body, re.M):
        sys.exit(f"{payload}: must define lab(deadline, weights) -> list[str]")
    return payload.stem, hashlib.sha256(body.encode()).hexdigest()[:16], body


def lab_payloads(lab: Path) -> list:
    """The (name, sha, source) payloads embedded in a built lab file."""
    for node in ast.parse(lab.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_LAB_PAYLOADS" for t in node.targets):
            return [tuple(e) for e in ast.literal_eval(node.value)]
    return []


def absorb_waiting(host: str, embedded: list, priority: float, note: str):
    """Fold a lab still waiting for a judge slot on this host into this send. okc_run sends under its
    dispatch lock, so holding the lock while stopping the waiter means it can't be sending meanwhile."""
    with (REPO / "runs/.dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for p in sorted((REPO / "runs/waiting").glob("*.json")):
            try:
                w = json.loads(p.read_text())
                os.kill(w["pid"], 0)
            except (OSError, ValueError, KeyError):
                continue
            lab = REPO / w["file"]
            if w["problem"] != host or lab.parent != REPO / "problems" / host / "lab" or w.get("until", 0) > time.time():
                continue
            names = {n for n, _, _ in embedded}
            kept = [e for e in lab_payloads(lab) if e[0] not in names]
            if not kept or len(kept) + len(embedded) > MERGE_MAX:
                continue
            os.kill(w["pid"], signal.SIGTERM)
            while p.exists():  # the waiter removes its entry as it exits
                time.sleep(0.1)
            print(f"merged the waiting lab {w['file']} ({', '.join(n for n, _, _ in kept)}) into this run", flush=True)
            return kept + embedded, min(priority, w["priority"]), f"{note} || {w.get('note', w['file'])}"
    return embedded, priority, note


def build(embedded: list, host: str, import_budget: float = IMPORT_BUDGET_S) -> tuple[str, str]:
    if not 1 <= len(embedded) <= MERGE_MAX:
        sys.exit(f"a lab needs 1 to {MERGE_MAX} payloads")
    if not 0 < import_budget < float("inf"):
        sys.exit("--import-budget must be positive and finite")
    if len({name for name, _, _ in embedded}) < len(embedded):
        sys.exit("payload names must differ (they label the receipt's sections)")
    base = best_source(host)
    src = (REPO / base["source"]).read_text()
    # Attention's judge never calls prepare. Its best source may still define a private,
    # zero-argument prepare() tuner that it calls itself during import; leave that intact.
    has_prepare = host != "muse_attention_decode" and any(
        isinstance(n, ast.FunctionDef) and n.name == "prepare" for n in ast.parse(src).body)
    out = src.rstrip() + "\n\n" + WRAPPER.format(
        names=", ".join(f"{n} {s}" for n, s, _ in embedded), base=base["job"], payloads=embedded,
        limit=LIMIT)
    out += (PREPARE_WRAPPER if has_prepare else IMPORT_WRAPPER).format(
        budget=300 - MARGIN_S if has_prepare else import_budget)
    return out, base["job"]


def read(ref: str, name: str | None = None):
    p = Path(ref)
    if not p.exists():
        hits = sorted(REPO.glob(f"runs/*/*_{ref}.json"))
        if not hits and (REPO / "runs/pending" / f"{ref}.json").exists():
            subprocess.run([sys.executable, str(REPO / "tools/okc_run.py"), "collect", ref], cwd=REPO, check=True)
            hits = sorted(REPO.glob(f"runs/*/*_{ref}.json"))
        if not hits:
            sys.exit(f"no receipt for {ref} (still running? `tools/okc_run.py collect {ref}`)")
        p = hits[-1]
    tail = json.loads(p.read_text())["result"].get("log_tail", "")
    parts = [m.group(0) for m in re.finditer(r"^LAB_BEGIN (\S+).*?(?:^LAB_END.*?$|\Z)", tail, re.M | re.S)
             if name is None or m.group(1) == name]
    print("\n".join(parts) if parts else f"no LAB_BEGIN{' ' + name if name else ''} in {p} (log_tail {len(tail)} chars):\n{tail[-2000:]}")


def quota() -> dict:
    """Today's (scored, check) runs left per problem from `okc problems`, or {} if okc can't say."""
    okc = shutil.which("okc") or str(Path.home() / ".okc" / "bin" / "okc")
    try:
        d = json.loads(subprocess.run([okc, "problems", "--json"], capture_output=True, text=True, timeout=60).stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    return {p["name"]: (p["limits"]["scored"]["left"], p["limits"]["checks"]["left"]) for p in d.get("problems", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("send", "build"):
        s = sub.add_parser(name)
        s.add_argument("payloads", type=Path, nargs="+")
        s.add_argument("--host", default="muse_bf16_proj", choices=HOSTS)
        s.add_argument("--import-budget", type=float, default=IMPORT_BUDGET_S,
                       help="payload deadline budget in seconds on hosts without prepare() (default: %(default)s)")
        if name == "send":
            s.add_argument("--note", required=True)
            s.add_argument("--priority", type=float, default=1.0)
            s.add_argument("--not-before", help="pass an ISO8601 send-time gate to okc_run; no early judge send")
        else:
            s.add_argument("-o", type=Path, required=True)
    r = sub.add_parser("read")
    r.add_argument("ref", help="job id or receipt .json")
    r.add_argument("name", nargs="?", help="only this payload's section")
    args = ap.parse_args()

    if args.cmd == "read":
        return read(args.ref, args.name)
    embedded = [embed(p) for p in args.payloads]
    if args.cmd == "build":
        src, base = build(embedded, args.host, args.import_budget)
        args.o.write_text(src)
        return print(f"{args.o}: {len(src)} bytes, base {base}")
    left = quota()
    if left and not sum(left.get(args.host, (0, 0))):
        spare = ", ".join(f"{h} {r}+{c}" for h, (r, c) in left.items() if h in HOSTS and r + c)
        print(f"WARNING: {args.host} has no runs left until 00:00 UTC, so this lab waits for the reset. "
              f"Hosts with scored+check runs left: {spare or 'none'}. A payload that needs only its own "
              f"synthetic data or the MoE experts can go on --host muse_moe_w4a16.", file=sys.stderr)
    note = f"h200_lab {' '.join(map(str, args.payloads))}: {args.note}"
    # An explicitly scheduled lab must not pull another lane's ready payload
    # into its delay. Other sends also skip this held entry via its until field.
    embedded, priority, note = (embedded, args.priority, note) if args.not_before else absorb_waiting(args.host, embedded, args.priority, note)
    src, base = build(embedded, args.host, args.import_budget)
    out = REPO / "problems" / args.host / "lab" / f"lab_{'+'.join(n for n, _, _ in embedded)}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    command=[sys.executable, str(REPO / "tools/okc_run.py"), "send", str(out.relative_to(REPO)),
             "--priority", str(priority), "--note", note]
    if args.not_before:command += ["--not-before", args.not_before]
    subprocess.run(command, cwd=REPO, check=False)


if __name__ == "__main__":
    main()
