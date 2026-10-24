"""H200 prefill expert sweep on the base source's prepared NVFP4 banks.

The third argument is supplied by tools/h200_lab.py and is the state that the
host tuner just prepared.  Keeping that state makes this a measurement payload,
not a second prepare pass.  Routes come from the layer's own router fixture,
including its busiest local EP8 expert group.
"""

import copy
import time

import torch


def lab(deadline, weights, state):
    state["_deadline"] = deadline
    state["_generator"] = torch.Generator(device=state["qkv"].device).manual_seed(478191)
    out = ["SWEEP host=muse_prefill_layer routing=router_realistic local_ep=8"]

    # Build exact-route Humming plans used by cand_mid_experts2.  The base source
    # already owns the transformed lossless bank; only the exact M plans are new.
    mid = None
    try:
        base = state["humming_variants"]["humming_masked"]
        mid = copy.copy(base)
        mid.tma = False
        mid.gemm_type = GemmType.GROUPED_MASKED
        mid.kernels = {"w13": {}, "w2": {}}
        mid.plans = {}
        mid.compile([753, 3012])
        if 753 in mid.plans:
            mid.plans[1024] = mid.plans[753]
        if 3012 in mid.plans:
            mid.plans[4096] = mid.plans[3012]
        out.append("BUILD humming_mid=ok")
    except BaseException as exc:
        out.append(f"BUILD humming_mid=NA:{type(exc).__name__}:{str(exc).splitlines()[0][:120]}")

    # V9 is a serving-side <=256-row control.  It is optional in the judge
    # image, so its absence is recorded rather than turning the lab into an
    # error.  The packed weights are the same tensors received by prefill.
    v9_mod = v9_state = None
    try:
        from sglang.srt.layers import muse_spark_moe as v9_mod
        # Prefill's checkpoint names the two W13 tensor scales as one fused
        # scale.  The serving V9 adapter accepts the equivalent split aliases.
        v9_weights = dict(weights)
        v9_weights["w13_gs_gate"] = weights.get("w13_gs_gate", weights["w13_gs"])
        v9_weights["w13_gs_up"] = weights.get("w13_gs_up", weights["w13_gs"])
        v9_weights["post_norm_w"] = weights.get("post_norm_w", weights["post_expert_norm"])
        v9_state = v9_mod.prepare(v9_weights)
        out.append("BUILD v9=ok")
    except BaseException as exc:
        out.append(f"BUILD v9=NA:{type(exc).__name__}:{str(exc).splitlines()[0][:120]}")

    def v9_chunks(data):
        if v9_mod is None or v9_state is None:
            raise RuntimeError("v9 unavailable")
        pooled, probs, slot_e, _slot_pos, _counts = data
        pieces = []
        # The serving V9 kernel is intentionally kept at its <=256-row bound.
        for lo in range(0, pooled.shape[0], 256):
            hi = min(lo + 256, pooled.shape[0])
            # Preserve remote routes as an out-of-range global ID; clamping
            # them to expert zero would inflate one local expert's M.
            ids = torch.where(slot_e[lo:hi] >= 0, slot_e[lo:hi],
                              torch.full_like(slot_e[lo:hi], 255)).to(torch.int32)
            pieces.append(v9_mod._v9_call(
                {"hidden_states": pooled[lo:hi], "topk_ids": ids,
                 "topk_weights": probs[lo:hi],
                 "config": {"local_count": 32, "d_hidden": 4096,
                            "d_expert": 4096, "local_start": v9_state["local_start"]}},
                v9_state, return_dense=True))
        return pieces[-1] if len(pieces) == 1 else pieces

    def run_variant(data, name):
        pooled, probs, slot_e, slot_pos, counts = data
        if name == "tma_cluster":
            raw = dict(state["raw_states"]["raw17"])
            raw["sms"] = int(raw["sms"]) * 4
            return _raw_experts_tiled(pooled, probs, slot_e, slot_pos, counts, raw)
        if name == "humming_mid":
            if mid is None:
                raise RuntimeError("humming_mid unavailable")
            hs = dict(state)
            hs["humming"] = mid
            return _experts_humming_masked(pooled, probs, slot_e, slot_pos, counts, hs)
        if name == "v9_le256":
            return v9_chunks(data)
        return _score_expert(pooled, probs, slot_e, slot_pos, counts, state, name)

    def timed(data, name):
        if time.monotonic() + 8.0 >= deadline:
            return None, "deadline"
        try:
            # One warmup compiles the selected graph and establishes the same
            # cache state for every contender; two events are enough for a lab
            # table while leaving room for all three prefill phases.
            run_variant(data, name)
            torch.cuda.synchronize()
            vals = []
            for _ in range(2):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                run_variant(data, name)
                b.record()
                b.synchronize()
                vals.append(a.elapsed_time(b) * 1000.0)
            vals.sort()
            return vals[0], "ok"
        except BaseException as exc:
            torch.cuda.synchronize()
            return None, f"{type(exc).__name__}:{str(exc).splitlines()[0][:120]}"

    # Keep the requested names explicit in the receipt.  dg_w8 and its hinted
    # sibling include their own in-kernel dequant; raw17 is the persistent TMA
    # producer/consumer path, while tma_cluster changes only its launch grid.
    names = ("raw17", "humming_masked", "dg_w8", "dg_hint_w8", "tma_cluster",
             "humming_mid", "v9_le256")
    for tokens in (753, 3012, 16384):
        if time.monotonic() + 8.0 >= deadline:
            out.append(f"PHASE T={tokens} skipped=deadline")
            continue
        try:
            data = _expert_realistic(tokens, state)
            route_total = int(data[4].sum().item())
            route_distinct = int((data[4] > 0).sum().item())
            out.append(f"ROUTE T={tokens} local_routes={route_total} experts={route_distinct}")
        except BaseException as exc:
            out.append(f"PHASE T={tokens} route=NA:{type(exc).__name__}:{str(exc).splitlines()[0][:120]}")
            continue
        for name in names:
            us, status = timed(data, name)
            out.append(f"RESULT T={tokens} variant={name} us={us if us is not None else 'NA'} status={status}")
        del data
        torch.cuda.empty_cache()
    return out

