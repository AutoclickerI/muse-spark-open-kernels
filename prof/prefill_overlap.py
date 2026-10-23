"""Local full-width Muse prefill/verify stream-overlap ceiling.

This is a TP8-rank *proxy*, not a full-model server: reuse one layer's
immutable weights for 46 sliding + 16 full graph replays. Collectives are
identity, and on non-Hopper only FA3 attention is replaced by cand_v13's
BF16 Triton implementation. All other operators come from baseline.py.
Each forward stream has a separate Marlin workspace, router side stream,
input/cache buffers, and private CUDA graph pools. No serving code is patched.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = ROOT / "problems/muse_prefill_layer"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def operators():
    sys.path.insert(0, str(PROBLEM))
    ref = load("overlap_ref", PROBLEM / "reference.py")
    if torch.cuda.get_device_capability() != (9, 0):
        portable = load("overlap_attention", PROBLEM / "cand_v13.py")

        def attention(**kw):
            q = kw["q"]
            batch = kw["cache_seqlens"].numel()
            length = kw["max_seqlen_q"]
            seq = kw["cache_seqlens"] - length
            scale = kw["softmax_scale"]
            window = kw["window_size"][0] + 1
            data = {
                "config": dict(tokens=q.shape[0], batch=batch, q_len=length,
                               window=window, page_size=kw["k_cache"].shape[1],
                               kv_scale=1.0, sm_scale=scale, qk_scale_factor=1.0),
                "k_cache": kw["k_cache"], "v_cache": kw["v_cache"],
                "page_table": kw["page_table"], "seq_lens": seq,
                "k_scale": scale_one, "v_scale": scale_one,
            }
            return portable._attention(data, q, {}).view(-1, 16, 64)

        scale_one = torch.ones((), device="cuda")
        stub = types.ModuleType("sgl_kernel.flash_attn")
        stub.flash_attn_with_kvcache = attention
        sys.modules["sgl_kernel.flash_attn"] = stub
    return ref, load("overlap_baseline", PROBLEM / "baseline.py")


def capture(base, data, state, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            base.kernel_fn(data, state)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = base.kernel_fn(data, state)
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    return graph, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, choices=(16, 32), required=True)
    ap.add_argument("--prefill-ctx", type=int, default=32768)
    ap.add_argument("--verify-ctx", type=int, default=80000)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--layers", choices=(1, 62), type=int, default=62)
    ap.add_argument("--burst", type=int, default=1,
                    help="number of verify steps per prefill chunk")
    ap.add_argument("--one-step-reps", type=int, default=0,
                    help="also time individual verify steps with these repetitions")
    ap.add_argument("--expert-backend", choices=("stock", "serving"), default="stock")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    ref, base = operators()
    print("Preparing full-width rank weights", flush=True)
    weights = ref.generate_weights(20261024)
    state = base.prepare(weights)
    port = None
    if args.expert_backend == "serving":
        path = ROOT / "serving/sglang/python/sglang/srt/layers/muse_spark_moe.py"
        port = load("overlap_serving_experts", path)
        expert_state = port.prepare({**weights, "w13_gs_gate": weights["w13_gs"],
                                     "w13_gs_up": weights["w13_gs"],
                                     "post_norm_w": weights["post_expert_norm"]})

        def experts(x, scores, ids, st):
            data = dict(hidden_states=x, topk_ids=ids,
                        topk_weights=scores, config=dict(local_count=32,
                        d_hidden=4096, d_expert=4096, local_start=0))
            dense, mapping = (port._grouped_call(data, expert_state) if x.shape[0] >= 753
                              else port._v9_call(data, expert_state, return_dense=True))
            local_ids = st["bank"]["expert_map"][ids]
            return base.fused_post_expert_norm_weighted_sum(
                dense, scores, norm_weight=st["post_expert_w"],
                eps=base.POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)

        base._experts = experts
        print(json.dumps({"serving_experts": str(path), "smem": {
            m: expert_state["mod"].smem_needed(1, 2, m) for m in (128, 256)},
            "resident_blocks": {m: expert_state["mod"].resident_blocks(1, 2, m)
                                for m in (128, 256)}}), flush=True)
    # Immutable parameters are shared; mutable workspaces never are.
    verify_state = {**state, "bank": {**state["bank"]},
                    "side_stream": torch.cuda.Stream(priority=-1)}
    verify_state["bank"]["workspace"] = torch.zeros_like(state["bank"]["workspace"])
    ps = torch.cuda.Stream(priority=0)
    ds = torch.cuda.Stream(priority=-1)
    graphs = {}
    outputs = {}
    inputs = {}
    for mode in ("prefill", "verify"):
        for layer in ("sliding", "full"):
            case = dict(batch=1 if mode == "prefill" else args.concurrency,
                        ctx=args.prefill_ctx if mode == "prefill" else args.verify_ctx,
                        q_len=16384 if mode == "prefill" else 8, layer=layer,
                        page=1, dist="soft", spread="fixed", seed=20261025)
            case.update(ref.case_fields(case, weights))
            data = ref.generate_input(**case)
            print(f"Capture {mode} {layer}: {case}", flush=True)
            key = mode, layer
            inputs[key] = data
            graphs[key], outputs[key] = capture(
                base, data, state if mode == "prefill" else verify_state,
                ps if mode == "prefill" else ds)
    flush = torch.empty(1 << 29, device="cuda", dtype=torch.uint8)
    def replay(mode):
        if args.layers == 1:
            graphs[mode, "full"].replay()
        else:
            for i in range(62):
                layer = "full" if i % 4 == 3 or i == 61 else "sliding"
                graphs[mode, layer].replay()

    def time_pair(mode, priority):
        # Graph replays can be issued on streams other than their capture stream.
        stream = ds if priority == -1 else normal_decode
        start, pe, de, end = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        flush.zero_()
        start.record()
        ps.wait_event(start)
        stream.wait_event(start)
        if mode in ("prefill", "serial", "concurrent"):
            with torch.cuda.stream(ps):
                replay("prefill")
                pe.record()
        if mode in ("verify", "serial", "concurrent"):
            if mode == "serial":
                stream.wait_event(pe)
            with torch.cuda.stream(stream):
                for _ in range(args.burst):
                    replay("verify")
                de.record()
        if mode in ("prefill", "serial", "concurrent"):
            torch.cuda.current_stream().wait_event(pe)
        if mode in ("verify", "serial", "concurrent"):
            torch.cuda.current_stream().wait_event(de)
        end.record()
        end.synchronize()
        return dict(total_ms=start.elapsed_time(end),
                    prefill_done_ms=start.elapsed_time(pe) if mode != "verify" else None,
                    verify_done_ms=start.elapsed_time(de) if mode != "prefill" else None)

    # Keep two ordinary-priority streams as well as the decode-priority pair.
    normal_decode = torch.cuda.Stream(priority=0)
    def measurements(reps):
        rows = []
        for rep in range(reps):
            modes = [("prefill", -1), ("verify", -1), ("serial", -1),
                     ("concurrent", -1), ("concurrent", 0)]
            if rep % 2:
                modes.reverse()
            for mode, priority in modes:
                sample = time_pair(mode, priority)
                row = dict(rep=rep, mode=mode, decode_priority=priority,
                           burst=args.burst, **sample)
                rows.append(row)
                print(json.dumps(row), flush=True)
        return rows

    rows = measurements(args.reps)
    saved_burst = args.burst
    args.burst = 1
    one_step_rows = measurements(args.one_step_reps)
    args.burst = saved_burst
    # Measure stock self-repeat spread before comparing concurrent execution.
    for data in inputs.values():
        data["attn_input"].add_(0.001)
        data["stream"].add_(0.001)
    def check_pair(concurrent):
        ps.wait_stream(torch.cuda.current_stream())
        ds.wait_stream(torch.cuda.current_stream())
        for mode, stream in (("prefill", ps), ("verify", ds)):
            if not concurrent:
                stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for layer in ("sliding", "full"):
                    graphs[mode, layer].replay()
            if not concurrent:
                torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.current_stream().wait_stream(ps)
        torch.cuda.current_stream().wait_stream(ds)
        torch.cuda.synchronize()

    check_pair(False)
    expected = {k: [v.clone() for v in val] for k, val in outputs.items()}
    kv_expected = {k: (data["k_cache"].view(torch.uint8).clone(),
                       data["v_cache"].view(torch.uint8).clone())
                   for k, data in inputs.items()}
    def differences():
        result = {}
        for k, val in outputs.items():
            result[str(k)] = []
            for x, y in zip(val, expected[k]):
                delta = (x.float() - y.float()).abs()
                result[str(k)].append(dict(exact=torch.equal(x, y),
                                          finite=bool(torch.isfinite(x).all()),
                                          max_abs=float(delta.max()),
                                          relative_l2=float(delta.norm() / y.float().norm().clamp_min(1e-30))))
        return result

    self_spread, concurrent_spread = [], []
    for _ in range(5):
        check_pair(False)
        self_spread.append(differences())
        check_pair(True)
        concurrent_spread.append(differences())
    self_errors, errors = self_spread[-1], concurrent_spread[-1]
    exact = all(row["exact"] for val in errors.values() for row in val)
    kv_exact = all(torch.equal(data[cache].view(torch.uint8), kv_expected[k][i])
                   for k, data in inputs.items()
                   for i, cache in enumerate(("k_cache", "v_cache")))
    def summarize(samples):
        summary = {}
        for mode, priority in (("prefill", -1), ("verify", -1), ("serial", -1),
                               ("concurrent", -1), ("concurrent", 0)):
            selected = [r for r in samples if r["mode"] == mode and r["decode_priority"] == priority]
            if selected:
                summary[f"{mode}:{priority}"] = {
                    key: statistics.median(r[key] for r in selected) if selected[0][key] is not None else None
                    for key in ("total_ms", "prefill_done_ms", "verify_done_ms")}
        return summary
    result = dict(args={**vars(args), "out": str(args.out)}, gpu=torch.cuda.get_device_name(),
                  capability=torch.cuda.get_device_capability(),
                  scope=f"shared-immutable-weight {args.layers}-layer replay proxy; no collectives; portable attention",
                  concurrent_vs_isolated_exact=exact, kv_bytes_exact=kv_exact,
                  output_errors=errors,
                  self_repeat_errors=self_errors,
                  self_spread=self_spread, concurrent_spread=concurrent_spread,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  nccl_version=torch.cuda.nccl.version(),
                  priorities=dict(prefill=ps.priority, verify=ds.priority),
                  sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), PROBLEM / "baseline.py",
                                     PROBLEM / "reference.py", PROBLEM / "cand_v13.py",
                                     *([path] if port is not None else []))},
                  summary=summarize(rows), samples=rows,
                  one_step_summary=summarize(one_step_rows),
                  one_step_samples=one_step_rows, timestamp=time.time())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2), flush=True)
    assert kv_exact and all(row["finite"] for val in errors.values() for row in val)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
