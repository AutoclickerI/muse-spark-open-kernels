#!/usr/bin/env python3
"""Record fixed-prompt and teacher-forced logprobs from one serving tree.

The request is deliberately made through the real HTTP endpoint.  The first
request obtains a greedy continuation; the second supplies that continuation
as part of the input and asks SGLang for ``input_token_logprobs`` from position
zero.  Thus the entries after the fixed prompt are the logprobs of the first
``k`` generated tokens, independent of speculative acceptance.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
import run  # noqa: E402


def request(port: int, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def token_logprobs(meta: dict) -> list[float | None]:
    return [None if row[0] is None else float(row[0]) for row in meta.get("input_token_logprobs", [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="stock")
    ap.add_argument("--weights", default="margin", choices=sorted(common.WEIGHT_SETS))
    ap.add_argument("--label", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=2600)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--attention-backend", default="triton")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()
    out = common.RESULTS / args.label
    if out.exists():
        ap.error(f"result exists: {out}")
    out.mkdir(parents=True)
    if args.gpus is None:
        args.gpus = "1" if args.tp == 1 else ",".join(str(i) for i in range(args.tp))
    target, draft = common.model_dirs(args.weights)
    opts = argparse.Namespace(
        target=str(target), draft=str(draft), tp=args.tp,
        mem_fraction_static=.85, max_total_tokens=300000,
        attention_backend=args.attention_backend, extra=args.extra,
        env=args.env, gpus=args.gpus,
    )
    proc, cmd = run.launch(opts, common.tree_for(args.tree), out, run.free_port())
    port = int(cmd[cmd.index("--port") + 1])
    prompt = run.build_workload(0, [1], {1: 1}, 1, [args.prompt_len])[0]["input_ids"]
    sampling = {"temperature": 0.0, "top_k": 1, "max_new_tokens": args.k,
                "ignore_eos": True}
    base_payload = {
        "input_ids": prompt, "sampling_params": sampling,
        "return_logprob": True, "logprob_start_len": 0,
        "top_logprobs_num": 0,
    }
    started = time.time()
    try:
        startup = run.wait_ready(proc, port, out / "server.log", 1200)
        base = request(port, base_payload)
        continuation = list(base.get("output_ids") or [])[:args.k]
        if len(continuation) != args.k:
            raise RuntimeError(f"base request returned {len(continuation)} tokens, expected {args.k}")
        forced_prompt = prompt + continuation
        forced_payload = {
            "input_ids": forced_prompt,
            "sampling_params": {"temperature": 0.0, "top_k": 1,
                                 "max_new_tokens": 1, "ignore_eos": True},
            "return_logprob": True, "logprob_start_len": 0,
            "top_logprobs_num": 0,
        }
        forced = request(port, forced_payload)
        input_lps = token_logprobs(forced.get("meta_info", {}))
        if len(input_lps) != len(forced_prompt):
            raise RuntimeError(f"forced input logprob length {len(input_lps)} != prompt length {len(forced_prompt)}")
        if input_lps[0] is not None:
            raise RuntimeError("first input logprob must be null")
        forced_tokens = forced.get("meta_info", {}).get("input_token_logprobs", [])
        forced_ids = [row[1] for row in forced_tokens]
        if forced_ids != forced_prompt:
            raise RuntimeError("server returned input logprobs in a different token order")
        record = {
            "label": args.label, "tree": args.tree, "weights": args.weights,
            "tp": args.tp, "prompt_len": len(prompt), "k": args.k,
            "prompt_ids": prompt, "generated_ids": continuation,
            "generated_output_logprobs": base.get("meta_info", {}).get("output_token_logprobs", []),
            "prompt_input_logprobs": input_lps,
            "forced_generated_token_logprobs": input_lps[len(prompt):len(prompt) + args.k],
            "forced_generated_token_ids": forced_ids[len(prompt):len(prompt) + args.k],
            "base_meta_info": base.get("meta_info", {}),
            "forced_meta_info": forced.get("meta_info", {}),
            "startup_s": startup, "wall_s": time.time() - started,
            "command": cmd,
        }
        (out / "teacher_forced.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({k: record[k] for k in (
            "label", "tree", "tp", "prompt_len", "k", "generated_ids",
            "forced_generated_token_logprobs")}, indent=2), flush=True)
    finally:
        run.stop(proc)
        if not (out / "teacher_forced.json").exists():
            (out / "failure.json").write_text(json.dumps({"label": args.label, "command": cmd}, indent=2))


if __name__ == "__main__":
    main()
