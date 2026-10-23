#!/usr/bin/env python3
"""Compare two harness runs request by request: token agreement, 220-token check, TTFT, acceptance.

    serving/harness/compare.py stock-a stock-b      # stock's self-agreement (the baseline)
    serving/harness/compare.py stock-a patch-1      # our patch against stock

Arguments are labels under the results dir or paths to run dirs. Exit status is 1 when either
run has a request that did not return exactly 220 tokens, an errored request, a server
traceback, or when the two runs did not replay the same workload.

Metrics per request (A = first run, B = second):
  exact       B's 220 ids equal A's
  prefix      tokens before the first difference, over 220 (1.0 when exact)
  positional  fraction of the 220 positions with equal ids (after a divergence, greedy outputs
              usually stay different, so this is mostly prefix plus chance matches)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def load(run: str) -> tuple[dict, dict]:
    p = Path(run)
    if not p.is_dir():
        p = common.RESULTS / run
    meta = json.loads((p / "meta.json").read_text())
    reqs = {}
    duplicates = []
    for line in (p / "requests.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r['key'] in reqs:
            duplicates.append(r['key'])
        reqs[r["key"]] = r
    meta['_duplicate_keys'] = duplicates
    return meta, reqs


def agreement(a: list[int], b: list[int]) -> dict:
    n = common.OUTPUT_TOKENS
    first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return {
        "exact": len(a) == n and len(b) == n and a == b,
        "first_diff": first,
        "prefix": min(first if first is not None else min(len(a), len(b)), n) / n,
        "positional": sum(1 for i in range(min(len(a), len(b), n)) if a[i] == b[i]) / n,
    }


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def compare(ma, ra, mb, rb) -> dict:
    keys = sorted(set(ra) & set(rb), key=lambda k: (ra[k]["level"], ra[k]["conv"], ra[k]["turn"]))
    rows = []
    for k in keys:
        a, b = ra[k], rb[k]
        row = {"key": k, "level": a["level"], **agreement(a.get("output_ids", []), b.get("output_ids", [])),
               "n_out_a": a["n_out"], "n_out_b": b["n_out"],
               "ttft_a": a.get("ttft"), "ttft_b": b.get("ttft"),
               "accept_a": a.get("spec_accept_length"), "accept_b": b.get("spec_accept_length"),
               "cached_a": a.get("cached_tokens"), "cached_b": b.get("cached_tokens")}
        rows.append(row)
    levels = sorted({r["level"] for r in rows})
    per_level = {}
    for lv in levels + (["all"] if rows else []):
        rs = [r for r in rows if lv == "all" or r["level"] == lv]
        per_level[str(lv)] = {
            "requests": len(rs),
            "exact": sum(r["exact"] for r in rs) / len(rs),
            "prefix_mean": statistics.mean(r["prefix"] for r in rs),
            "prefix_min": min(r["prefix"] for r in rs),
            "positional_mean": statistics.mean(r["positional"] for r in rs),
            "first_diff_median": median([r["first_diff"] for r in rs if r["first_diff"] is not None]),
            "all_220_a": all(r["n_out_a"] == common.OUTPUT_TOKENS for r in rs),
            "all_220_b": all(r["n_out_b"] == common.OUTPUT_TOKENS for r in rs),
            "ttft_p50_a": median([r["ttft_a"] for r in rs]),
            "ttft_p50_b": median([r["ttft_b"] for r in rs]),
            "accept_mean_a": statistics.mean([r["accept_a"] for r in rs if r["accept_a"]] or [0]),
            "accept_mean_b": statistics.mean([r["accept_b"] for r in rs if r["accept_b"]] or [0]),
        }
        pl = per_level[str(lv)]
        pl["ttft_ratio_b_over_a"] = (pl["ttft_p50_b"] / pl["ttft_p50_a"]
                                     if pl["ttft_p50_a"] and pl["ttft_p50_b"] else None)
        pl['ttft_p50_within_2x'] = (pl['ttft_ratio_b_over_a'] <= 2
                                   if pl['ttft_ratio_b_over_a'] is not None else None)
    problems = []
    warnings = []
    if not rows:
        problems.append('no paired requests')
    if ma["workload"]["sha256"] != mb["workload"]["sha256"]:
        problems.append("different workloads (seed/levels/convs/turns differ)")
    if ma.get('checkpoint_sha256') and mb.get('checkpoint_sha256'):
        if ma['checkpoint_sha256'] != mb['checkpoint_sha256']:
            problems.append('different checkpoint contents')
    else:
        warnings.append('legacy result lacks checkpoint content fingerprints')
    if ma.get('sampling_params') != mb.get('sampling_params'):
        problems.append('different sampling parameters')
    if ma['tp'] != mb['tp']:
        problems.append('different TP sizes')
    for field in ('attention_backend', 'draft_attention_backend'):
        if ma.get('log', {}).get(field) != mb.get('log', {}).get(field):
            problems.append(f'different resolved {field}')
    for name, m, r in (("A", ma, ra), ("B", mb, rb)):
        bad = [k for k, v in r.items() if v["n_out"] != common.OUTPUT_TOKENS
               or len(v.get('output_ids', [])) != common.OUTPUT_TOKENS]
        errs = [k for k, v in r.items() if "error" in v]
        if len(r) != m['workload']['requests']:
            problems.append(f"{name}: expected {m['workload']['requests']} requests, got {len(r)}")
        if m.get('_duplicate_keys'):
            problems.append(f"{name}: duplicate request keys")
        for audit in m.get('load_check', {}).values():
            if audit.get('unwritten_or_partial'):
                problems.append(f'{name}: weight loading audit failed')
                break
        if bad:
            problems.append(f"{name}: {len(bad)} requests without exactly {common.OUTPUT_TOKENS} tokens: {bad[:5]}")
        if errs:
            problems.append(f"{name}: {len(errs)} errored requests: {errs[:5]}")
        if m.get("failure"):
            problems.append(f"{name}: run failed: {m['failure']}")
        problems += [f'{name}: {error}' for error in m.get('validation_errors', [])]
        if m.get("log", {}).get("tracebacks"):
            problems.append(f"{name}: {m['log']['tracebacks']} tracebacks in server.log")
    missing = sorted(set(ra) ^ set(rb))
    if missing:
        problems.append(f"{len(missing)} requests present in only one run: {missing[:5]}")
    return {"a": ma["label"], "b": mb["label"], "per_level": per_level, "rows": rows,
            "problems": problems, "warnings": warnings}


def fmt(x, spec=".3f"):
    return "-" if x is None else format(x, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", type=Path, help="also write the full comparison here")
    ap.add_argument("--rows", action="store_true", help="print every request")
    ap.add_argument('--baseline', nargs=2, metavar=('STOCK_A', 'STOCK_B'),
                    help='require a meaningful stock baseline and report agreement delta')
    ap.add_argument('--require-exact', action='store_true', help='fail on any differing output token')
    args = ap.parse_args()
    ma, ra = load(args.a)
    mb, rb = load(args.b)
    res = compare(ma, ra, mb, rb)
    if args.baseline:
        m0, r0 = load(args.baseline[0]); m1, r1 = load(args.baseline[1])
        base = compare(m0, r0, m1, r1)
        if (m0['workload']['sha256'] != ma['workload']['sha256'] or
                m0.get('checkpoint_sha256') != ma.get('checkpoint_sha256')):
            res['problems'].append('baseline uses a different workload or checkpoint')
        res['baseline'] = {k: v for k, v in base.items() if k != 'rows'}
        res['problems'] += [f'baseline: {p}' for p in base['problems']]
        for lv, metric in res['per_level'].items():
            bs = base['per_level'].get(lv, {})
            metric['positional_delta_vs_self'] = metric['positional_mean'] - bs.get('positional_mean', 0)
            if bs.get('positional_mean', 0) < 0.5:
                res['problems'].append(f'baseline at concurrency {lv} is too unstable to validate agreement')
    if args.require_exact and any(not row['exact'] for row in res['rows']):
        res['problems'].append('output token disagreement (--require-exact)')
    print(f"A = {res['a']}  ({ma['provenance']['tree']}, diff sha {ma['provenance']['diff_vs_base_sha256'][:12]})")
    print(f"B = {res['b']}  ({mb['provenance']['tree']}, diff sha {mb['provenance']['diff_vs_base_sha256'][:12]})")
    hdr = (f"{'conc':>5} {'reqs':>4} {'exact':>6} {'prefix':>7} {'pfx_min':>7} {'posit':>6} {'1st_diff':>8}"
           f" {'220 A/B':>8} {'ttft A':>7} {'ttft B':>7} {'B/A':>5} {'acc A':>6} {'acc B':>6}")
    print(hdr)
    for lv, s in res["per_level"].items():
        print(f"{lv:>5} {s['requests']:>4} {fmt(s['exact'], '.2f'):>6} {fmt(s['prefix_mean']):>7}"
              f" {fmt(s['prefix_min']):>7} {fmt(s['positional_mean']):>6} {fmt(s['first_diff_median'], '.0f'):>8}"
              f" {('ok' if s['all_220_a'] else 'NO') + '/' + ('ok' if s['all_220_b'] else 'NO'):>8}"
              f" {fmt(s['ttft_p50_a']):>7} {fmt(s['ttft_p50_b']):>7} {fmt(s['ttft_ratio_b_over_a'], '.2f'):>5}"
              f" {fmt(s['accept_mean_a'], '.2f'):>6} {fmt(s['accept_mean_b'], '.2f'):>6}")
    if args.rows:
        for r in res["rows"]:
            print(f"  {r['key']:<20} exact={r['exact']!s:<5} first_diff={r['first_diff']!s:<4}"
                  f" n={r['n_out_a']}/{r['n_out_b']} cached={r['cached_a']}/{r['cached_b']}"
                  f" accept={fmt(r['accept_a'], '.2f')}/{fmt(r['accept_b'], '.2f')}")
    for p in res["problems"]:
        print("PROBLEM:", p)
    for p in res['warnings']:
        print('WARNING:', p)
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
    sys.exit(1 if res["problems"] else 0)


if __name__ == "__main__":
    main()
