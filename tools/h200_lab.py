#!/usr/bin/env python3
"""Batch H200 lab payloads on any problem host.

    tools/h200_lab.py send probe.py [more.py ...] --host muse_prefill_layer --lab-only --import-budget 700 --lane P4 --decision "promote composition" --note "fresh pairs"
    tools/h200_lab.py build probe.py [more.py ...] -o /tmp/lab.py
    tools/h200_lab.py read <job | receipt.json> [name]

Payload: lab(deadline, weights[, state]) returns {"packet": evidence, "logs": lines},
or the existing list[str] interface. Evidence contains identities, seeds, paired, checks,
complete. The wrapper emits a lossless checksummed LAB_PACKET after every payload has
finished; on import hosts it re-emits after host prepare. read recovers existing result
lines; programmatic consumers use tools.lab_packet.result_lines/read_packets.

Declare a literal LAB_RESOURCES = {"compile_s": 30, "run_s": 40, "output_bytes": 2000,
"context_poison": "none"}. output_bytes includes the encoded packet line and newline;
256 additional bytes per payload cover section markers. Batches fit declared time and
reserved output first; diagnostics use remaining space. Device-poison probes run alone;
"last" probes run last. Undeclared existing payloads reserve a whole output allocation.
The CLI splits batches automatically (build writes name.py, name.2.py, ...). The Python
build API creates exactly one batch and rejects a batch needing a split.

Prepare phase uses the best scored host's unchanged prepare/state. --base-prepare gives
its estimated cost for planning (default 130 s); the real deadline is prepare start+285 s.
--phase import runs with weights/state None after the base definitions. --lab-only uses
a minimal import host and skips the scored source's compile/prepare entirely; it
deliberately stops at prepare after emitting evidence. Non-decode hosts spend spare
scored quota first (then checks); use --check to preserve scored quota. Decode labs
always use checks. Payloads own setup and honor deadline. lab_cold_l2 reads a buffer larger than L2, as TASK prescribes.

The judge keeps a 16,000-character tail. Ordinary transports reserve 9,000 wire bytes
for packets/markers, leaving 7,000 characters for profiler/error suffixes. --lab-only
reserves 12,000: its deliberate prepare stop skips profiling and leaves 4,000 for errors. Queue rank comes from the
PLAN lane of the decision, independently of the host used for quota and transport.
"""

import argparse
import ast
import fcntl
import hashlib
import json
import math
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
from okc_packaging import local_imports  # noqa: E402
from lab_packet import read_packets, result_lines  # noqa: E402

LIMIT = 9000          # packet wire bytes + bounded diagnostics + markers; 7000 chars for judge suffix
LAB_ONLY_LIMIT = 12000  # no timed/profile phase; leave 4000 chars for its bounded prepare-stop suffix
MARGIN_S = 15         # left of prepare's 300 s after the payload's deadline
HOSTS = ("muse_bf16_proj", "muse_moe_w4a16", "muse_prefill_layer", "muse_decode_layer", "muse_attention_decode")
IMPORT_BUDGET_S = 150  # H200 probe 33d8bb6861b64c9e completed 150 s after the host's own ~45 s import lab
BASE_PREPARE_S = 130
# Undeclared payloads get a whole packet allocation. Declare LAB_RESOURCES to batch safely.
DEFAULT_RESOURCES = dict(compile_s=30, run_s=40, output_bytes=LIMIT-256, context_poison="none")


def resources(entry):
    value = dict(DEFAULT_RESOURCES)
    for n in ast.parse(entry[2]).body:
        if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'LAB_RESOURCES' for t in n.targets):
            value.update(ast.literal_eval(n.value))
    for key in ('compile_s', 'run_s'):
        if not isinstance(value[key], (int, float)) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError(f"{entry[0]}: {key} must be finite and nonnegative")
    if value['compile_s'] + value['run_s'] <= 0:
        raise ValueError(f"{entry[0]}: compile_s + run_s must be positive")
    if not isinstance(value['output_bytes'], int) or not 512 <= value['output_bytes'] <= LIMIT-256:
        raise ValueError(f"{entry[0]}: output_bytes must be 512..{LIMIT-256} wire bytes")
    if value['context_poison'] not in ('none', 'last', 'device'):
        raise ValueError(f"{entry[0]}: context_poison must be none, last or device")
    return value


def plan_batches(embedded, budget_s, output_budget=LIMIT):
    """Pack time and reserved wire output; context-poisoning probes get their own run."""
    if not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError('batch budget must be positive and finite')
    if len({e[0] for e in embedded}) != len(embedded):
        raise ValueError('payload names must differ')
    batches = []
    for entry in embedded:
        if not re.fullmatch(r'[\w.+-]{1,80}', entry[0]):
            raise ValueError('payload name must be 1..80 letters, numbers, _, ., + or -')
        res = resources(entry)
        need = res['compile_s'] + res['run_s']
        if need > budget_s:
            raise ValueError(f"{entry[0]} needs {need:g}s; only {budget_s:g}s declared available; use import/lab-only or increase --import-budget")
        for batch in batches:
            rr = [resources(e) for e in batch]
            if (res['context_poison'] == 'device' or any(r['context_poison'] == 'device' for r in rr)
                    or sum(r['context_poison'] == 'last' for r in rr) + (res['context_poison'] == 'last') > 1):
                continue
            if (sum(r['compile_s']+r['run_s'] for r in rr)+need <= budget_s
                    and sum(r['output_bytes']+256 for r in rr)+res['output_bytes']+256 <= output_budget):
                batch.append(entry)
                batch.sort(key=lambda e: resources(e)['context_poison'] == 'last')
                break
        else:
            batches.append([entry])
    return batches

WRAPPER = '''
# ---- h200_lab wrapper: payloads {names} on base {base} ----
_LAB_PAYLOADS = {payloads!r}
_lab_ran = False
_lab_flush_buf = None
_lab_output = []


def _lab_print(line):
    {print_line}


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
        _records = []
        _reserve = sum(r["output_bytes"] + 256 for r in _LAB_RESOURCES)
        _log_cap = max(0, {limit} - _reserve) // len(_LAB_PAYLOADS)
        for _i, (_name, _sha, _src) in enumerate(_LAB_PAYLOADS):
            _t0 = _t.monotonic()
            _res = _LAB_RESOURCES[_i]
            _need = _res["compile_s"] + _res["run_s"]
            _remaining = sum(r["compile_s"]+r["run_s"] for r in _LAB_RESOURCES[_i:])
            _deadline = min(_end, _t0 + max(_need, max(0., _end-_t0)*_need/_remaining))
            _lines = []
            _packet = None
            try:
                if _t0 >= _end:
                    raise TimeoutError("lab budget exhausted before this payload")
                _result = _lab_run_one(_name, _src, _deadline, weights, state)
                if isinstance(_result, dict):
                    _packet = _result["packet"]
                    if not all(k in _packet for k in ("identities", "seeds", "paired", "checks", "complete")):
                        raise ValueError("packet needs identities, seeds, paired, checks, complete")
                    _lines = [str(x) for x in _result.get("logs", [])]
                else:
                    _lines = [str(x) for x in (_result or [])]
                    _packet = dict(lines=_lines, complete=None, format="legacy-lines")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except BaseException as _e:
                _errors = [f"LAB_ERROR {{type(_e).__name__}}: {{_e}}"] + _tb.format_exc().splitlines()[-8:]
                _lines += _errors
                if _packet is None:
                    _packet = dict(lines=_lines, complete=False)
                else:
                    _packet = dict(_packet, complete=False, transport_errors=_errors)
            try:
                _wire = _lab_encode_packet(_name, _sha, _packet)
            except Exception as _e:
                _wire = _lab_encode_packet(_name, _sha, dict(complete=False, lines=[
                    f"LAB_PACKET_ERROR {{type(_e).__name__}}: {{str(_e)[:100]}} "]))
            if len(_wire.encode()) + 1 > _res["output_bytes"]:
                # Never cut a packet. The planner needs a larger declaration or a smaller experiment.
                _wire = _lab_encode_packet(_name, _sha, dict(complete=False, lines=[
                    f"LAB_PACKET_OVERSIZE needs {{len(_wire.encode())+1}} bytes; declared {{_res['output_bytes']}}; split payload"]))
            _diagnostics = []
            _used = 0
            for _line in _lines:
                if _used + len(_line.encode()) + 1 <= _log_cap:
                    _diagnostics.append(_line)
                    _used += len(_line.encode()) + 1
            _records.append([f"LAB_BEGIN {{_name}} {{_sha}}"] + _diagnostics + [_wire,
                f"LAB_END lab_s={{_t.monotonic()-_t0:.1f}} {{_phase}}_s={{_t.monotonic()-_start:.1f}}"])
        # Emit all reserved packets after every payload (and, on import hosts, after base prepare).
        # A later compiler's stdout cannot evict an earlier packet from the judge's receipt tail.
        for _record in _records:
            for _line in _record:
                _lab_print(_line)
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
# Weight generation can fail before the judge calls host prepare (for example
# after a failed capture). Emit evidence now and retain it for the prepare replay.
for _lab_early_line in _lab_output:
    print(_lab_early_line, flush=True)
'''

IMPORT_PREPARE_WRAPPER = '''
_lab_base_prepare = prepare


def prepare(weights):
    try:
        return _lab_base_prepare(weights)
    finally:
        # Emit after the host tuner, whose output can exceed the judge's entire log tail.
        for _line in _lab_output:
            print(_line, flush=True)
        _lab_output.clear()
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


def absorb_waiting(host: str, embedded: list, priority: float, note: str, *, budget_s=155, config=None, lane=None, decision=None):
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
            if lane is not None and (w.get('lane') != lane or w.get('decision') != decision):
                continue
            lab = REPO / w["file"]
            if w["problem"] != host or lab.parent != REPO / "problems" / host / "lab" or w.get("until", 0) > time.time():
                continue
            # Merge only the exact queued snapshot and the same transport/base identity.
            if hashlib.sha256(lab.read_bytes()).hexdigest() != w['sha256']:
                continue
            old_config = next((ast.literal_eval(n.value) for n in ast.parse(lab.read_text()).body
                               if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == '_LAB_CONFIG' for t in n.targets)), None)
            if config is None or config != old_config:
                continue
            names = {n for n, _, _ in embedded}
            kept = [e for e in lab_payloads(lab) if e[0] not in names]
            if not kept or len(plan_batches(kept + embedded, budget_s, config['output_budget'])) != 1:
                continue
            import select
            try:
                fd = os.pidfd_open(w['pid'])
            except ProcessLookupError:
                p.unlink(missing_ok=True)
                continue
            try:
                os.kill(w['pid'], signal.SIGTERM)
                select.select([fd], [], [])  # process completion, not a polling cadence
            finally:
                os.close(fd)
            p.unlink(missing_ok=True)
            print(f"merged the waiting lab {w['file']} ({', '.join(n for n, _, _ in kept)}) into this run", flush=True)
            return kept + embedded, min(priority, w["priority"]), f"{note} || {w.get('note', w['file'])}"
    return embedded, priority, note


def build(embedded: list, host: str, import_budget: float = IMPORT_BUDGET_S, phase: str = "auto", base_file: Path | None = None, lab_only: bool = False, base_prepare_s: float = BASE_PREPARE_S) -> tuple[str, str]:
    if not embedded:
        sys.exit("a lab needs at least one payload")
    if not 0 < import_budget < float("inf"):
        sys.exit("--import-budget must be positive and finite")
    if len({name for name, _, _ in embedded}) < len(embedded):
        sys.exit("payload names must differ (they label the receipt's sections)")
    if lab_only and base_file:
        sys.exit("--lab-only supplies its own host; omit --base-file")
    if lab_only:
        base = {"job": "lab-only " + host}
        src = (f"# okc: problem={host}\nimport torch\n"
               "def prepare(weights):raise RuntimeError('lab-only host: evidence completed at import')\n"
               "def kernel_fn(*args):raise RuntimeError('lab-only host')\n"
               "kernel=kernel_fn\n")
        phase = "import"
    else:
        base = best_source(host) if base_file is None else {"source": str(base_file), "job": "candidate " + str(base_file)}
        src = (REPO / base["source"]).read_text()
    if not re.match(r"#\s*okc:\s*problem=" + re.escape(host) + r"\b", src):
        sys.exit("lab base must target --host")
    # Attention's judge never calls prepare. Its best source may still define a private,
    # zero-argument prepare() tuner that it calls itself during import; leave that intact.
    if phase not in ("auto", "import"):
        sys.exit("--phase must be auto or import")
    has_prepare = host != "muse_attention_decode" and any(
        isinstance(n, ast.FunctionDef) and n.name == "prepare" for n in ast.parse(src).body)
    prepare_phase = has_prepare and phase != "import"
    budget = import_budget if not prepare_phase else 300 - MARGIN_S - base_prepare_s
    output_budget = LAB_ONLY_LIMIT if lab_only else LIMIT
    batches = plan_batches(embedded, budget, output_budget)
    if len(batches) != 1:
        raise ValueError("batch exceeds declared time/output/risk; split with plan_batches (the CLI does this automatically)")
    embedded = batches[0]
    config = dict(host=host, phase='prepare' if prepare_phase else 'import', import_budget=import_budget,
                  base_sha256=hashlib.sha256(src.encode()).hexdigest(), lab_only=lab_only, base_prepare_s=base_prepare_s, output_budget=output_budget)
    helpers = (REPO / 'tools/lab_packet.py').read_text()
    out = src.rstrip() + "\n\n_LAB_CONFIG = " + repr(config) + "\n_LAB_RESOURCES = " + repr([resources(e) for e in embedded])
    out += "\n_lab_packet_ns = {}\nexec(" + repr(helpers) + ", _lab_packet_ns)\n_lab_encode_packet = _lab_packet_ns['encode_packet']\n"
    out += WRAPPER.format(
        names=", ".join(f"{n} {s}" for n, s, _ in embedded), base=base["job"], payloads=embedded,
        limit=output_budget, print_line="_lab_output.append(line)" if has_prepare and not prepare_phase else "print(line, flush=True)")
    out += (PREPARE_WRAPPER if prepare_phase else IMPORT_WRAPPER).format(
        budget=300 - MARGIN_S if prepare_phase else import_budget)
    if has_prepare and not prepare_phase:
        out += IMPORT_PREPARE_WRAPPER
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
    if packets := read_packets(tail):
        for record in packets:
            if name is None or record['name'] == name:
                packet = record['packet']
                print(f"LAB_BEGIN {record['name']} {record['source_sha256']}")
                print("\n".join(packet['lines']) if 'lines' in packet else json.dumps(packet, indent=2))
                print('LAB_END checksum=' + record['checksum'])
        return
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
        s.add_argument("--phase", choices=("auto", "import"), default="auto",
                       help="force import-time labs when payloads own setup; host prepare remains unchanged")
        s.add_argument("--base-file", type=Path, help="explicit screened candidate to validate as the lab host")
        s.add_argument("--lab-only", action="store_true", help="minimal import-time host; skips the scored host's import/prepare")
        s.add_argument("--base-prepare", type=float, default=BASE_PREPARE_S, metavar="SECONDS",
                       help="declared host prepare cost for batch planning (default: 130)")
        if name == "send":
            s.add_argument("--note", required=True)
            s.add_argument("--priority", type=float, help="legacy override without --lane")
            s.add_argument("--lane", help="PLAN decision lane, independent of host (P4, P10, ...)")
            s.add_argument("--decision", default="", help="one-line decision for the ledger")
            s.add_argument("--not-before", help="pass an ISO8601 send-time gate to okc_run; no early judge send")
            s.add_argument("--check", action="store_true", help="reserve scored quota; send only as a check "
                           "(always on muse_decode_layer, whose scored runs carry candidates)")
        else:
            s.add_argument("-o", type=Path, required=True)
    r = sub.add_parser("read")
    r.add_argument("ref", help="job id or receipt .json")
    r.add_argument("name", nargs="?", help="only this payload's section")
    args = ap.parse_args()

    if args.cmd == "read":
        return read(args.ref, args.name)
    embedded = [embed(p) for p in args.payloads]
    if not math.isfinite(args.base_prepare) or args.base_prepare < 0:
        ap.error('--base-prepare must be finite and nonnegative')
    if args.lab_only and args.base_file:
        ap.error('--lab-only supplies its own host; omit --base-file')
    if not args.lab_only and args.base_file is None:
        args.base_file = REPO / best_source(args.host)['source']
    has_prepare = (not args.lab_only and args.host != 'muse_attention_decode' and any(
        isinstance(n, ast.FunctionDef) and n.name == 'prepare' for n in ast.parse(args.base_file.read_text()).body))
    prepare_phase = has_prepare and args.phase != 'import'
    budget = 300 - MARGIN_S - args.base_prepare if prepare_phase else args.import_budget
    batches = plan_batches(embedded, budget, LAB_ONLY_LIMIT if args.lab_only else LIMIT)
    if args.cmd == "build":
        for i, batch in enumerate(batches):
            src, base = build(batch, args.host, args.import_budget, args.phase, args.base_file,
                              args.lab_only, args.base_prepare)
            path = args.o if i == 0 else args.o.with_name(args.o.stem + f'.{i+1}' + args.o.suffix)
            path.write_text(src)
            print(f"{path}: {len(src)} bytes, base {base}, batch {i+1}/{len(batches)}")
        return
    left = quota()
    if left and not sum(left.get(args.host, (0, 0))):
        spare = ", ".join(f"{h} {r}+{c}" for h, (r, c) in left.items() if h in HOSTS and r + c)
        print(f"WARNING: {args.host} has no runs left until 00:00 UTC, so this lab waits for the reset. "
              f"Hosts with scored+check runs left: {spare or 'none'}. A payload that needs only its own "
              f"synthetic data or the MoE experts can go on --host muse_moe_w4a16.", file=sys.stderr)
    note = f"h200_lab {' '.join(map(str, args.payloads))}: {args.note}"
    from okc_run import queue_metadata
    metadata = queue_metadata(args.lane, args.decision, note=args.note, priority=args.priority)
    for batch in batches:
        priority = metadata['priority']
        batch_note = note
        # An explicitly scheduled lab never absorbs a ready lane into its delay.
        if not args.not_before:
            probe, _ = build(batch, args.host, args.import_budget, args.phase, args.base_file,
                             args.lab_only, args.base_prepare)
            config = next(ast.literal_eval(n.value) for n in ast.parse(probe).body if isinstance(n, ast.Assign)
                          and any(getattr(t, 'id', None) == '_LAB_CONFIG' for t in n.targets))
            batch, priority, batch_note = absorb_waiting(args.host, batch, priority, batch_note,
                                                        budget_s=budget, config=config, lane=metadata["lane"], decision=metadata["decision"])
        src, base = build(batch, args.host, args.import_budget, args.phase, args.base_file,
                          args.lab_only, args.base_prepare)
        out = REPO / "problems" / args.host / "lab" / f"lab_{'+'.join(n for n, _, _ in batch)}.py"
        out.parent.mkdir(exist_ok=True)
        out.write_text(src)
        command=[sys.executable, str(REPO / "tools/okc_run.py"), "send", str(out.relative_to(REPO)),
                 "--note", batch_note, "--decision", metadata['decision']]
        if args.lane:command += ["--lane", args.lane]
        elif args.priority is not None:command += ["--priority", str(priority)]
        elif metadata["lane"] != "unspecified":command += ["--lane", metadata["lane"]]
        if args.not_before:command += ["--not-before", args.not_before]
        if args.check or args.host == "muse_decode_layer":command += ["--check"]
        subprocess.run(command, cwd=REPO, check=True)



if __name__ == "__main__":
    main()
