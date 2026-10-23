"""Measure target-layer row batching, with separate causal attention segments.

This is the same local rank proxy as prefill_overlap.py, not a serving patch.
Dense, routing, experts and norms run once on [prefill; verify] rows. The
portable attention stand-in runs the two disjoint request groups separately.
Native FA3 already accepts the resulting ragged query offsets on Hopper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from prefill_overlap import PROBLEM, ROOT, capture, load, operators


def mixed_data(p, d):
    length = p["config"]["q_len"]
    count = d["config"]["batch"]
    offset = p["k_cache"].shape[0]
    width = max(p["page_table"].shape[1], d["page_table"].shape[1])
    result = {**p}
    for name in ("attn_input", "stream", "positions", "seq_lens", "cache_seqlens"):
        result[name] = torch.cat((p[name], d[name]))
    result["out_cache_loc"] = None
    result["cache_loc"] = torch.cat((p["cache_loc"], d["cache_loc"] + offset))
    for cache in ("k_cache", "v_cache"):
        result[cache] = torch.cat((p[cache], d[cache]))
    dp = torch.where(d["page_table"] >= 0, d["page_table"] + offset, -1)
    result["page_table"] = torch.cat((
        F.pad(p["page_table"], (0, width - p["page_table"].shape[1]), value=-1),
        F.pad(dp, (0, width - dp.shape[1]), value=-1)))
    result["cu_seqlens_q"] = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32),
        length + torch.arange(count + 1, device="cuda", dtype=torch.int32) * 8))
    result["scheduler_metadata"] = None
    result["config"] = {**p["config"], "tokens": length + count * 8, "batch": count + 1}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, choices=(16, 32), required=True)
    ap.add_argument("--prefill-length", type=int, choices=(753, 16384), required=True)
    ap.add_argument("--layers", type=int, choices=(1, 62), default=62)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--expert-backend", choices=("stock", "serving"), default="stock")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    ref, base = operators()
    uniform = base.flash_attn_with_kvcache
    if torch.cuda.get_device_capability() != (9, 0):
        def attention(**kw):
            if kw["max_seqlen_q"] == args.prefill_length and kw["cache_seqlens"].numel() > 1:
                left, right = dict(kw), dict(kw)
                for k in ("cache_seqlens", "page_table"):
                    left[k], right[k] = kw[k][:1], kw[k][1:]
                left["q"], right["q"] = kw["q"][:args.prefill_length], kw["q"][args.prefill_length:]
                right["max_seqlen_q"] = 8
                return torch.cat((uniform(**left), uniform(**right)))
            return uniform(**kw)
        base.flash_attn_with_kvcache = attention
    weights = ref.generate_weights(20261024)
    state = base.prepare(weights)
    producer_path = None
    if args.expert_backend == "serving":
        producer_path = ROOT / "serving/sglang/python/sglang/srt/layers/muse_spark_moe.py"
        port = load("mixed_serving_experts", producer_path)
        expert_state = port.prepare({**weights, "w13_gs_gate": weights["w13_gs"],
                                     "w13_gs_up": weights["w13_gs"],
                                     "post_norm_w": weights["post_expert_norm"]})

        def experts(x, scores, ids, st):
            payload = dict(hidden_states=x, topk_ids=ids, topk_weights=scores,
                           config=dict(local_count=32, d_hidden=4096,
                                       d_expert=4096, local_start=0))
            dense, _ = (port._grouped_call(payload, expert_state) if x.shape[0] >= 753
                        else port._v9_call(payload, expert_state, return_dense=True))
            return base.fused_post_expert_norm_weighted_sum(
                dense, scores, norm_weight=st["post_expert_w"], eps=base.POST_NORM_EPS,
                weight_first=True, slot_expert_ids=st["bank"]["expert_map"][ids])

        base._experts = experts
    stream = torch.cuda.Stream()
    graphs, data, outputs = {}, {}, {}
    for layer in ("sliding", "full"):
        for mode in ("prefill", "verify"):
            case = dict(batch=1 if mode == "prefill" else args.concurrency,
                        q_len=args.prefill_length if mode == "prefill" else 8,
                        ctx=(32768 if args.prefill_length == 16384 else 80000)
                        if mode == "prefill" else 80000,
                        layer=layer, page=1, dist="soft", spread="fixed", seed=20261025)
            if mode == "prefill":
                case.update(ref.case_fields(case, weights))
                local_start = case["local_start"]
            else:
                case["local_start"] = local_start
            data[mode, layer] = ref.generate_input(**case)
        data["mixed", layer] = mixed_data(data["prefill", layer], data["verify", layer])
        for mode in ("prefill", "verify", "mixed"):
            print(f"Capture {mode} {layer}", flush=True)
            graphs[mode, layer], outputs[mode, layer] = capture(base, data[mode, layer], state, stream)
    flush = torch.empty(1 << 29, device="cuda", dtype=torch.uint8)

    def replay(mode):
        for i in range(args.layers):
            layer = "full" if args.layers == 1 or i % 4 == 3 or i == 61 else "sliding"
            graphs[mode, layer].replay()

    def measure(mode):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        flush.zero_()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            start.record()
            if mode == "serial":
                replay("prefill")
                replay("verify")
            else:
                replay(mode)
            end.record()
        end.synchronize()
        return start.elapsed_time(end)

    rows = []
    for rep in range(args.reps):
        order = ("prefill", "verify", "serial", "mixed")
        if rep % 2:
            order = tuple(reversed(order))
        for mode in order:
            row = dict(rep=rep, mode=mode, ms=measure(mode))
            rows.append(row)
            print(json.dumps(row), flush=True)
    errors = {}
    for layer in ("sliding", "full"):
        errors[layer] = []
        for i, value in enumerate(outputs["mixed", layer]):
            expected = torch.cat((outputs["prefill", layer][i], outputs["verify", layer][i]))
            delta = value.float() - expected.float()
            errors[layer].append(dict(exact=torch.equal(value, expected),
                finite=bool(torch.isfinite(value).all()), max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-30))))
    diagnostics = {}
    if args.diagnostics:
        original_gemm = base._gemm
        for layer in ("sliding", "full"):
            stages = {}
            for mode in ("prefill", "verify", "mixed"):
                recorded = []

                def traced_gemm(x, w):
                    out = original_gemm(x, w)
                    recorded.append((x.clone(), out.clone()))
                    return out

                base._gemm = traced_gemm
                base.kernel_fn(data[mode, layer], state)
                torch.cuda.synchronize()
                stages[mode] = recorded
            base._gemm = original_gemm
            diagnostics[layer] = []
            for i, name in enumerate(("qkv", "output_projection", "pooler", "unpooler")):
                for part, label in ((0, "input"), (1, "output")):
                    expected = torch.cat((stages["prefill"][i][part], stages["verify"][i][part]))
                    value = stages["mixed"][i][part]
                    delta = value.float() - expected.float()
                    per_group = []
                    for group in (slice(None, args.prefill_length), slice(args.prefill_length, None)):
                        per_group.append(float(delta[group].norm() / expected[group].float().norm().clamp_min(1e-30)))
                    diagnostics[layer].append(dict(stage=name, part=label, exact=torch.equal(value, expected),
                                                   prefill_relative_l2=per_group[0], verify_relative_l2=per_group[1]))
    summary = {mode: statistics.median(r["ms"] for r in rows if r["mode"] == mode)
               for mode in ("prefill", "verify", "serial", "mixed")}
    result = dict(args={**vars(args), "out": str(args.out)}, gpu=torch.cuda.get_device_name(),
        scope="target-layer replay proxy; shared immutable rank weights; no TP collectives; portable attention",
        summary_ms=summary, samples=rows, output_errors=errors, diagnostics=diagnostics, timestamp=time.time(),
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (Path(__file__), ROOT / "prof/prefill_overlap.py", PROBLEM / "baseline.py",
                           *([producer_path] if producer_path else []))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2), flush=True)
    assert all(r["finite"] for val in errors.values() for r in val)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
