#!/usr/bin/env python3
"""Collect teacher-forced logprobs for every request in a saved workload run.

The reference run supplies one common prompt and continuation per request.  Each
tree is started once and is asked to score that same continuation, so the
resulting per-request drift is independent of whether greedy generation chose
the same token after a near-tie.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request

import common
import run


def load_run(name: str):
    path = pathlib.Path(name)
    if not path.is_dir():
        path = common.RESULTS / name
    workload = [json.loads(line) for line in (path / "workload.jsonl").read_text().splitlines()]
    requests = {
        row["key"]: row
        for row in (json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines())
    }
    return path, workload, requests


def request(port: int, input_ids: list[int], timeout: float) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def finite_logprobs(response: dict, prompt_len: int, forced_len: int) -> list[float]:
    rows = response.get("meta_info", {}).get("input_token_logprobs", [])
    if len(rows) != prompt_len + forced_len:
        raise RuntimeError(
            f"input logprob length {len(rows)} != {prompt_len + forced_len}"
        )
    ids = [row[1] for row in rows]
    if any(row[0] is None for row in rows[1:]):
        raise RuntimeError("unexpected null input logprob after the first token")
    return [float(row[0]) for row in rows[prompt_len:prompt_len + forced_len]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="stock")
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--weights", default="random", choices=sorted(common.WEIGHT_SETS))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    args = parser.parse_args()
    out = common.RESULTS / args.label
    if out.exists():
        parser.error(f"result exists: {out}")
    out.mkdir(parents=True)

    reference_path, workload, reference_requests = load_run(args.reference)
    if args.gpus is None:
        args.gpus = "1" if args.tp == 1 else ",".join(str(i) for i in range(args.tp))
    target, draft = common.model_dirs(args.weights)
    launch_args = argparse.Namespace(
        target=str(target),
        draft=str(draft),
        tp=args.tp,
        mem_fraction_static=0.85,
        max_total_tokens=300000,
        attention_backend=args.attention_backend,
        extra=[],
        env=args.env,
        gpus=args.gpus,
    )
    proc, command = run.launch(
        launch_args, common.tree_for(args.tree), out, run.free_port()
    )
    port = int(command[command.index("--port") + 1])
    started = time.time()
    records = []
    try:
        startup = run.wait_ready(proc, port, out / "server.log", 1200)
        for item in workload:
            key = item["key"]
            reference = reference_requests.get(key)
            if reference is None or len(reference.get("output_ids", [])) != common.OUTPUT_TOKENS:
                raise RuntimeError(f"reference lacks a complete 220-token response for {key}")
            prompt = item["input_ids"]
            forced = list(reference["output_ids"])
            response = request(port, prompt + forced, args.request_timeout)
            logprobs = finite_logprobs(response, len(prompt), len(forced))
            records.append({
                "key": key,
                "level": item["level"],
                "conv": item["conv"],
                "turn": item["turn"],
                "prompt_len": len(prompt),
                "forced_ids": forced,
                "logprobs": logprobs,
            })
            print(f"  {key:<20} forced={len(logprobs)}", flush=True)
        (out / "forced.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        summary = {
            "label": args.label,
            "tree": args.tree,
            "reference": str(reference_path),
            "requests": len(records),
            "startup_s": startup,
            "wall_s": time.time() - started,
            "command": command,
        }
        (out / "meta.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        run.stop(proc)
        if not (out / "forced.jsonl").exists():
            (out / "failure.json").write_text(
                json.dumps({"label": args.label, "command": command}, indent=2)
            )


if __name__ == "__main__":
    main()
