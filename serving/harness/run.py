#!/usr/bin/env python3
"""Serve the tiny Muse model from one SGLang tree with the judge's flags and record a workload.

Launches `python -m sglang.launch_server` inside the judge's image (serving/harness/img) with
<tree>/python first on PYTHONPATH, waits for it, replays deterministic multi-turn conversations
at each concurrency (exactly 220 output tokens each, greedy), saves every request's output ids
and latencies, then stops the server.

    serving/harness/run.py --tree stock --label stock-a
    serving/harness/run.py --tree patch --label patch-1
    serving/harness/compare.py stock-a patch-1

Runs on the host's python3 (stdlib only); only the server runs in the image.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# The judge's launch (upstream/api/serving.json "launch"). Model paths, parallel sizes and the
# memory fraction are filled in per run; everything else is passed through unchanged.
JUDGE_FLAGS = [
    "--quantization", "modelopt_fp4",
    "--moe-runner-backend", "marlin",
    "--speculative-algorithm", "DFLASH",
    "--speculative-num-draft-tokens", "8",
    "--enable-hierarchical-cache",
    "--chunked-prefill-size", "16384",
    "--context-length", "131072",
    "--enable-cache-report",
    "--incremental-streaming-output",
    "--random-seed", "1234",
    "--max-running-requests", "32",
    "--cuda-graph-max-bs-decode", "32",
]
JUDGE_ENV = {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "SGLANG_TIMEOUT_KEEP_ALIVE": "1800"}

ROLE_USER, ROLE_ASSISTANT = 1, 2          # role words inside the header tokens


# ------------------------------------------------------------------ workload
def build_workload(seed: int, levels: list[int], convs_per_level: dict[int, int], turns: int,
                   first_turn_lens: list[int]) -> list[dict]:
    """Deterministic multi-turn conversations of random word tokens.

    Turn k's prompt is the whole history (fixed assistant replies, not the model's own, so every
    run sees identical prompts) plus a new user message, and the assistant header. Consecutive
    turns share their prefix, so the radix/hierarchical cache is exercised as in the judge's
    agentic conversations. One first turn per level is longer than the 16,384-token chunked
    prefill and every one is longer than the 2,048-token window.
    """
    out = []
    for level in levels:
        for c in range(convs_per_level[level]):
            rng = random.Random(f"{seed}/{level}/{c}")

            def words(n):
                return [rng.randrange(common.NUM_WORD_TOKENS) for _ in range(n)]

            def message(role, n):
                return ([common.HEADER_START_ID, role, common.HEADER_END_ID] + words(n)
                        + [common.EOT_ID])

            history = [common.BOS_ID] + message(ROLE_USER, first_turn_lens[c % len(first_turn_lens)])
            for t in range(turns):
                if t > 0:
                    history += message(ROLE_ASSISTANT, common.OUTPUT_TOKENS)  # a fixed reply
                    history += message(ROLE_USER, rng.randrange(250, 800))
                prompt = history + [common.HEADER_START_ID, ROLE_ASSISTANT, common.HEADER_END_ID]
                out.append({"level": level, "conv": c, "turn": t, "key": f"c{level}/conv{c}/turn{t}",
                            "input_ids": prompt})
    return out


# ------------------------------------------------------------------ client
def stream_generate(port: int, input_ids: list[int], timeout: float) -> dict:
    body = json.dumps({
        "input_ids": input_ids,
        "stream": True,
        "sampling_params": {"temperature": 0.0, "top_k": 1, "max_new_tokens": common.OUTPUT_TOKENS,
                            "ignore_eos": True},
    })
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/generate", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    ids: list[int] = []
    chunks, ttft, meta = 0, None, {}
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        msg = json.loads(data)
        new = msg.get("output_ids") or []
        if new and ttft is None:
            ttft = time.perf_counter() - t0
        ids.extend(new)
        chunks += 1
        meta = msg.get("meta_info", meta)
    e2e = time.perf_counter() - t0
    conn.close()
    return {"output_ids": ids, "ttft": ttft, "e2e": e2e, "chunks": chunks,
            "completion_tokens": meta.get("completion_tokens"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
            "finish_reason": meta.get("finish_reason"),
            "spec_accept_length": meta.get("spec_accept_length"),
            'spec_num_correct_drafts': meta.get('spec_num_correct_drafts'),
            'spec_num_proposed_drafts': meta.get('spec_num_proposed_drafts'),
            'spec_accept_rate': meta.get('spec_accept_rate'),
            "spec_verify_ct": meta.get("spec_verify_ct"),
            "spec_correct_drafts_histogram": meta.get("spec_correct_drafts_histogram")}


def run_level(port: int, level: int, items: list[dict], timeout: float, on_result=None) -> list[dict]:
    """`level` workers, each taking whole conversations (turns in order) from a shared queue."""
    by_conv: dict[int, list[dict]] = {}
    for it in items:
        by_conv.setdefault(it["conv"], []).append(it)
    queue = [sorted(v, key=lambda x: x["turn"]) for _, v in sorted(by_conv.items())]
    lock = threading.Lock()
    results: list[dict] = []
    t_start = time.perf_counter()

    def worker(slot):
        while True:
            with lock:
                if not queue:
                    return
                conv = queue.pop(0)
            for it in conv:
                rec = {k: it[k] for k in ("level", "conv", "turn", "key")}
                rec["prompt_len"] = len(it["input_ids"])
                rec["slot"] = slot
                rec["t_send"] = time.perf_counter() - t_start
                try:
                    rec.update(stream_generate(port, it["input_ids"], timeout))
                except Exception as e:  # noqa: BLE001 - recorded, the run continues
                    rec["error"] = repr(e)
                rec["n_out"] = len(rec.get("output_ids", []))
                if 'error' not in rec and (rec['n_out'] != common.OUTPUT_TOKENS
                        or rec.get('completion_tokens') != common.OUTPUT_TOKENS
                        or rec.get('prompt_tokens') != rec['prompt_len']
                        or (rec.get('finish_reason') or {}).get('type') != 'length'):
                    rec['error'] = 'response length, prompt length, or finish reason failed validation'
                with lock:
                    results.append(rec)
                    if on_result is not None:
                        on_result(rec)
                    print(f"  {rec['key']:<20} prompt {rec['prompt_len']:>6}  out {rec['n_out']:>3}"
                          f"  ttft {rec.get('ttft') or float('nan'):6.3f}s  e2e {rec.get('e2e') or float('nan'):6.2f}s"
                          f"  accept {rec.get('spec_accept_length') or 0:4.2f}"
                          f"  cached {rec.get('cached_tokens')}" + (f"  ERROR {rec['error']}" if 'error' in rec else ""),
                          flush=True)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(level)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return sorted(results, key=lambda r: (r["conv"], r["turn"]))


# ------------------------------------------------------------------ server
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_ok(port: int, path: str, timeout: float = 5.0) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", path)
        ok = conn.getresponse().status == 200
        conn.close()
        return ok
    except OSError:
        return False


def tree_provenance(tree: Path) -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(tree), *a], capture_output=True, text=True).stdout
    diff = git("diff", common.BASE_COMMIT, "--", "python/sglang")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "python/sglang").split()
    return {"tree": str(tree), "head": git("rev-parse", "HEAD").strip(),
            "diff_vs_base_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "diff_vs_base_lines": diff.count("\n"), "untracked_files": untracked}


def launch(args, tree: Path, out: Path, port: int) -> tuple[subprocess.Popen, list[str]]:
    target = Path(args.target).resolve()
    draft = Path(args.draft).resolve()
    server_args = [
        "--model-path", str(target), "--speculative-draft-model-path", str(draft),
        "--tp-size", str(args.tp), "--ep-size", str(args.tp),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--max-total-tokens", str(args.max_total_tokens),
        *JUDGE_FLAGS,
        "--port", str(port), "--host", "127.0.0.1",
    ]
    if args.attention_backend:
        server_args += ["--attention-backend", args.attention_backend]
    server_args += args.extra
    env_args = [f"{k}={v}" for k, v in JUDGE_ENV.items()]
    # The tree under test first; pyhook/ supplies the image's prebuilt Rust extensions.
    env_args += [f"PYTHONPATH={tree / 'python'}:{common.HARNESS / 'pyhook'}",
                 "PYTHONDONTWRITEBYTECODE=1"] + list(args.env)
    cmd = [str(common.HARNESS / "img"), "env", *env_args, "python", "-m", "sglang.launch_server",
           *server_args]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            cwd=str(common.REPO), start_new_session=True)
    proc.harness_ready = threading.Event()
    proc.harness_fired = False

    def record_log():
        with open(out / "server.log", "wb") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if b"The server is fired up" in line:
                    proc.harness_fired = True
                    proc.harness_ready.set()
        proc.wait()
        proc.harness_ready.set()  # process exit is also a completion signal

    proc.harness_logger = threading.Thread(target=record_log, daemon=True)
    proc.harness_logger.start()
    return proc, cmd


def stop(proc: subprocess.Popen) -> None:
    # The launcher may have exited while a scheduler still owns a CUDA context.
    # Its session/process group remains ours until those children have exited.
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def wait_ready(proc, port: int, log: Path, timeout: float) -> float:
    t0 = time.monotonic()
    if not proc.harness_ready.wait(timeout):
        raise TimeoutError(f"server not ready after {timeout:.0f}s; see {log}")
    if not proc.harness_fired or proc.poll() is not None:
        raise RuntimeError(f"server exited with {proc.poll()}; see {log}")
    if not http_ok(port, "/health"):
        raise RuntimeError(f"server announced readiness but /health failed; see {log}")
    return time.monotonic() - t0


def scan_log(log: Path) -> dict:
    text = log.read_text(errors="replace")
    m = re.search(r"'attention_backend': '([^']+)'", text)
    d = re.search(r"'speculative_draft_attention_backend': '([^']+)'", text)
    t = re.search(r"Engine startup timings \(s\): (.*)", text)
    tc = re.search(r"Tree Core: (\w+)", text)
    return {
        "tree_core": tc.group(1) if tc else None,
        "attention_backend": m.group(1) if m else None,
        "draft_attention_backend": d.group(1) if d else None,
        "startup_timings": t.group(1).strip() if t else None,
        "tracebacks": text.count("Traceback (most recent call last)"),
        "error_lines": [ln[:300] for ln in text.splitlines()
                        if re.search(r"\b(ERROR|Error|error)\b", ln) and "error_lines" not in ln][:50],
        "nan_mentions": len(re.findall(r"\bnan\b|NaN", text)),
        'capture_completed': {name: len(re.findall(f'Capture {name} CUDA graph end', text))
                              for name in ('target prefill', 'target verify', 'draft verify')},
        'prefill_graph_replayed': bool(re.search(r'Prefill batch,.*cuda graph: True', text)),
        'decode_graph_replayed': bool(re.search(r'Decode batch,.*cuda graph: True', text)),
        'fp8_kv_pools': text.count('dtype: torch.float8_e4m3fn'),
    }


# Built by the model but deliberately absent from the tiny checkpoint (text-only workload).
EXPECTED_UNLOADED = ("model.vision_tower.", "model.vision_adapter.", "model.vision_projection.",
                     "model.audio_tower.")


def load_check_summary(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        rep = json.loads(f.read_text())
        bad = {n: r["unwritten_fraction"] for n, r in rep.items()
               if r["unwritten_fraction"] != 0.0 and not n.startswith(EXPECTED_UNLOADED)}
        expected = sum(1 for n in rep if n.startswith(EXPECTED_UNLOADED))
        out[f.stem] = {"params": len(rep), "unwritten_or_partial": bad,
                       "expected_unloaded(vision/audio)": expected}
    return out


def graph_errors(log: dict, tp: int, extra: list[str]) -> list[str]:
    """Detect automatic graph fallbacks as well as explicit capture errors."""
    errors = []
    if '--disable-cuda-graph' not in extra:
        for stage in ('target verify', 'draft verify'):
            if log['capture_completed'][stage] < tp:
                errors.append(f'{stage} CUDA graph capture missing on one or more ranks')
        if not log['decode_graph_replayed']:
            errors.append('no decode graph replay observed')
    if '--disable-prefill-cuda-graph' not in extra and '--disable-cuda-graph' not in extra:
        if log['capture_completed']['target prefill'] < tp or not log['prefill_graph_replayed']:
            errors.append('target prefill graph capture/replay missing')
    return errors


def summarize(results: list[dict], levels: list[int]) -> dict:
    def pct(xs, q):
        xs = sorted(x for x in xs if x is not None)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
    s = {}
    for lv in levels:
        rs = [r for r in results if r["level"] == lv]
        acc = [r["spec_accept_length"] for r in rs if r.get("spec_accept_length")]
        s[str(lv)] = {
            "requests": len(rs),
            "errors": sum("error" in r for r in rs),
            "all_220": bool(rs) and all(r["n_out"] == common.OUTPUT_TOKENS for r in rs),
            "n_out_not_220": [r["key"] for r in rs if r["n_out"] != common.OUTPUT_TOKENS],
            "ttft_p50": pct([r.get("ttft") for r in rs], 0.5),
            "ttft_p90": pct([r.get("ttft") for r in rs], 0.9),
            "e2e_p50": pct([r.get("e2e") for r in rs], 0.5),
            "mean_accept_length": sum(acc) / len(acc) if acc else None,
            "tokens_per_s_per_user_p50": pct([common.OUTPUT_TOKENS / (r["e2e"] - r["ttft"])
                                             for r in rs if r.get("ttft") and r.get("e2e")], 0.5),
        }
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default="patch", help="'stock', 'patch' (serving/sglang) or a path to an SGLang checkout")
    ap.add_argument("--label", required=True, help=f"results go to {common.RESULTS}/<label>")
    ap.add_argument("--weights", default="margin", choices=sorted(common.WEIGHT_SETS),
                    help="tiny checkpoint pair written by make_tiny.py (margin is the agreement default; random is a tie diagnostic)")
    ap.add_argument("--target", default=None, help="override the target checkpoint dir")
    ap.add_argument("--draft", default=None, help="override the drafter checkpoint dir")
    ap.add_argument("--tp", type=int, default=1, help="TP size; EP is set equal, as in the judge's flags")
    ap.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES (default: '1' for TP1, '0,1' for TP2)")
    ap.add_argument("--levels", default="1,4", help="concurrency levels, run in this order")
    ap.add_argument("--convs", default="1:2,4:8", help="conversations per level, 'level:n,...'")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--first-turn-lens", default="2600,5200,18000,3900",
                    help="first-turn user message lengths, cycled over conversations")
    ap.add_argument("--seed", type=int, default=0, help="workload seed (keep equal across runs you compare)")
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--max-total-tokens", type=int, default=300000)
    ap.add_argument("--attention-backend", default=None,
                    help="override (default: SGLang's choice, flashinfer on sm_120; the judge's H200 picks fa3)")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the server")
    ap.add_argument("--load-check", action="store_true",
                    help="report every target/drafter parameter SGLang's loader did not write (pyhook/)")
    ap.add_argument("--trace-paths", action="store_true", help="observe model calls and row-split returns")
    ap.add_argument("--startup-timeout", type=float, default=1200)
    ap.add_argument("--request-timeout", type=float, default=600)
    ap.add_argument("extra", nargs="*", help="extra server flags after --")
    args = ap.parse_args()
    target, draft = common.model_dirs(args.weights)
    args.target = args.target or str(target)
    args.draft = args.draft or str(draft)
    if args.gpus is None:
        args.gpus = "1" if args.tp == 1 else ",".join(str(i) for i in range(args.tp))

    tree = common.tree_for(args.tree)
    if not (tree / "python" / "sglang").is_dir():
        sys.exit(f"{tree} is not an SGLang checkout (no python/sglang)")
    provenance = tree_provenance(tree)
    if args.tree == 'stock' and (provenance['head'] != common.BASE_COMMIT or
            provenance['diff_vs_base_lines'] or provenance['untracked_files']):
        sys.exit('the stock tree must be pristine at ' + common.BASE_COMMIT)
    out = common.RESULTS / args.label
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists; pick a new --label")
    out.mkdir(parents=True)

    levels = [int(x) for x in args.levels.split(",")]
    convs = {int(k): int(v) for k, v in (p.split(":") for p in args.convs.split(","))}
    if not levels or any(lv < 1 or convs.get(lv, 0) < lv for lv in levels) or args.turns < 1:
        ap.error('each concurrency needs at least that many conversations, and turns >= 1')
    workload = build_workload(args.seed, levels, convs, args.turns,
                              [int(x) for x in args.first_turn_lens.split(",")])
    if args.load_check:
        args.env.append(f"MUSE_HARNESS_LOAD_CHECK={out / 'load_check'}")
    if args.trace_paths:
        args.env.append(f"MUSE_HARNESS_TRACE={out / 'trace'}")
    port = free_port()
    meta = {"label": args.label, "argv": sys.argv, "provenance": provenance,
            "weights": args.weights, "target": args.target, "draft": args.draft, "tp": args.tp, "gpus": args.gpus,
            "checkpoint_sha256": {"target": common.checkpoint_fingerprint(Path(args.target)),
                                  "draft": common.checkpoint_fingerprint(Path(args.draft))},
            "sampling_params": {"temperature": 0.0, "top_k": 1,
                                "max_new_tokens": common.OUTPUT_TOKENS, "ignore_eos": True},
            "workload": {"seed": args.seed, "levels": levels, "convs": convs, "turns": args.turns,
                         "first_turn_lens": args.first_turn_lens, "requests": len(workload),
                         "sha256": hashlib.sha256(json.dumps([w["input_ids"] for w in workload]).encode()).hexdigest()},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta['extra_server_args'] = args.extra
    meta['extra_environment'] = args.env
    (out / 'workload.jsonl').write_text(''.join(json.dumps(w) + '\n' for w in workload))
    (out / 'requests.jsonl').touch()
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f"[{args.label}] tree={tree} gpus={args.gpus} tp={args.tp} port={port} -> {out}", flush=True)
    t_run = time.time()
    proc, cmd = launch(args, tree, out, port)
    meta["server_cmd"] = cmd
    results: list[dict] = []

    def persist_result(record):
        # Close after each completed request: evidence survives an interrupted
        # controller, not just a clean final shutdown.
        with (out / 'requests.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
    try:
        meta["startup_s"] = wait_ready(proc, port, out / "server.log", args.startup_timeout)
        print(f"[{args.label}] server ready in {meta['startup_s']:.0f}s", flush=True)
        by_level = {lv: [w for w in workload if w["level"] == lv] for lv in levels}
        meta["level_wall_s"] = {}
        for lv in levels:
            print(f"[{args.label}] concurrency {lv}: {len(by_level[lv])} requests", flush=True)
            t0 = time.time()
            results += run_level(port, lv, by_level[lv], args.request_timeout, persist_result)
            meta["level_wall_s"][str(lv)] = time.time() - t0
            if proc.poll() is not None:
                raise RuntimeError(f"server died during concurrency {lv}")
    except Exception as e:  # noqa: BLE001 - recorded in meta.json, then re-raised
        meta["failure"] = repr(e)
        raise
    finally:
        stop(proc)
        results = [json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines()]
        proc.harness_logger.join(timeout=10)
        if proc.harness_logger.is_alive():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.harness_logger.join(timeout=10)
        meta['provenance_end'] = tree_provenance(tree)
        if meta['provenance_end'] != meta['provenance']:
            meta['failure'] = 'source tree changed during the run'
        meta["total_s"] = time.time() - t_run
        meta["log"] = scan_log(out / "server.log")
        meta['validation_errors'] = graph_errors(meta['log'], args.tp, args.extra)
        meta["summary"] = summarize(results, levels)
        if args.load_check:
            meta["load_check"] = load_check_summary(out / "load_check")
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({"summary": meta["summary"], "startup_s": meta.get("startup_s"),
                          "total_s": round(meta["total_s"]), "log": {k: meta["log"][k] for k in
                          ("attention_backend", "draft_attention_backend", "tree_core", "tracebacks")},
                          "load_check": meta.get("load_check"),
                          'validation_errors': meta['validation_errors'],
                          "failure": meta.get("failure")}, indent=1), flush=True)
    bad = (bool(meta.get('failure')) or bool(meta['validation_errors'])
           or len(results) != len(workload) or any('error' in r for r in results)
           or meta['log']['tracebacks'] > 0)
    if args.load_check:
        bad |= (len(meta['load_check']) < 2 * args.tp or
                any(v['unwritten_or_partial'] for v in meta['load_check'].values()))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
