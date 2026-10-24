# H200 lab payload: T=16384 DG-W8 expert decomposition on the real router fixture.
# It runs inside the already-prepared prefill host source and retains only timing lines.

def lab(deadline, weights, state):
    import statistics
    import time
    import deep_gemm
    T = 16384
    state["_generator"] = torch.Generator(device=state["qkv"].device).manual_seed(478191)
    dev = state["qkv"].device
    lines = ["DECOMP_BEGIN T=16384 variant=dg_w8"]
    flush = torch.empty(1 << 30, dtype=torch.uint8, device=dev)

    def measure(label, fn, reps=5):
        # Warm compilation and use cold-L2 event samples; the flush is outside events.
        try:
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            vals = []
            for _ in range(reps):
                _fused_flush_l2(flush)
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                b.synchronize()
                vals.append(a.elapsed_time(b) * 1000.0)
            med = statistics.median(vals)
            lines.append(f"TIME {label} us={med:.3f} samples=" + ",".join(f"{v:.1f}" for v in vals))
            return med
        except BaseException as exc:
            lines.append(f"TIME {label} us=NA error={type(exc).__name__}:{str(exc).splitlines()[0][:160]}")
            return None

    try:
        data = _expert_realistic(T, state)
        pooled, probs, slot_e, slot_pos, counts = data
        count_list = [int(x) for x in counts.detach().cpu().tolist()]
        padded = [((x + 127) // 128) * 128 for x in count_list]
        active = [x for x in count_list if x]
        lines.append("ROUTE local_routes=%d touched=%d counts=%s" %
                     (sum(count_list), len(active), ",".join(map(str, count_list))))
        lines.append("ROUTE_PADDED rows=%d max=%d min_active=%d mean_active=%.1f" %
                     (sum(padded), max(count_list), min(active), sum(active) / len(active)))

        starts = torch.empty(32, dtype=torch.int32, device=dev)
        psum = torch.empty_like(starts)
        _offsets_kernel[(1,)](counts, starts, psum, ALIGN=128)
        cap = T * 8 + 32 * 128
        a_in = torch.empty((cap, 4096), dtype=torch.bfloat16, device=dev)
        row_of = torch.empty((T, 8), dtype=torch.int32, device=dev)
        measure("pre_norm_gather", lambda: _gather_norm_kernel[(T,)](
            pooled, state["pre_expert"], slot_e, slot_pos, starts, a_in, row_of,
            D=4096, num_warps=8), reps=4)

        # DG-W8's exact path: 1024-K dequant into BF16, grouped W13, SwiGLU,
        # 1024-K dequant into BF16, grouped W2, then the routed combine.
        scratch = torch.empty((32, 8192, 4096), dtype=torch.bfloat16, device=dev)
        w2 = scratch.view(-1)[:32 * 4096 * 4096].view(32, 4096, 4096)
        h = torch.empty((cap, 8192), dtype=torch.bfloat16, device=dev)
        y = h.view(-1)[:cap * 4096].view(cap, 4096)
        inter = a_in
        kb = 1024
        measure("dequant_w13", lambda: _canonical_expand_coalesced[
            (32, 512, 4096 // kb)](
                state["w13p"], state["w13p"].view(torch.int32), scratch.view(torch.int32), counts,
                ROWS=8192, GATED=True, KB=kb, num_warps=8), reps=4)
        hint = {}
        measure("gemm_w13", lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            inter, scratch, h, psum, use_psum_layout=True, **hint), reps=4)
        measure("swiglu", lambda: _portfolio_swiglu[
            (32, triton.cdiv(T, 4))](h, inter, starts, counts,
                                     state["gs13_gate"], state["gs13_up"], D=4096,
                                     BR=4, num_warps=8), reps=4)
        measure("dequant_w2", lambda: _canonical_expand_coalesced[
            (32, 256, 4096 // kb)](
                state["w2p"], state["w2p"].view(torch.int32), w2.view(torch.int32), counts,
                ROWS=4096, GATED=False, KB=kb, num_warps=8), reps=4)
        measure("gemm_w2", lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            inter, w2, y, psum, use_psum_layout=True, **hint), reps=4)
        out = torch.empty_like(pooled)
        measure("combine", lambda: _combine_kernel[(T,)](
            y, row_of, slot_e, probs, state["gs2"], state["post_expert"], out,
            D=4096, num_warps=8), reps=4)
        measure("dg_w8_total", lambda: _score_expert(
            pooled, probs, slot_e, slot_pos, counts, state, "dg_w8"), reps=3)

        # Register-A paths have no BF16 weight bank; retain their total as a control
        # so the receipt can identify whether their loss is dequant, scheduling, or tail.
        for name in ("raw17", "raw_v11"):
            if time.monotonic() + 20.0 >= deadline:
                lines.append(f"CONTROL {name}=skipped_deadline")
                continue
            try:
                val = measure(f"control_{name}_total", lambda n=name: _score_expert(
                    pooled, probs, slot_e, slot_pos, counts, state, n), reps=3)
                lines.append(f"CONTROL {name}_us={val if val is not None else 'NA'}")
            except BaseException as exc:
                lines.append(f"CONTROL {name}=NA:{type(exc).__name__}:{str(exc).splitlines()[0][:160]}")
    except BaseException as exc:
        lines.append(f"DECOMP_ERROR {type(exc).__name__}:{str(exc).splitlines()[0][:240]}")
    finally:
        del flush
        torch.cuda.empty_cache()
    lines.append("DECOMP_END")
    return lines
