"""Tail bundled by build_lab.py, after prefixed V9 and router fixture helpers."""


def _bf_expand(packed, scale):
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=packed.device)
    result = torch.empty((*packed.shape[:-1], packed.shape[-1]*2), dtype=torch.bfloat16, device=packed.device)
    for e in range(packed.shape[0]):
        code = torch.stack((packed[e]&15, packed[e]>>4), -1).flatten(-2)
        values = grid[(code&7).long()] * torch.where(code&8 != 0, -1., 1.)
        values = values.view(*scale[e].shape, 16)*scale[e].float().unsqueeze(-1)
        result[e] = values.flatten(-2).to(torch.bfloat16)
    return result


def _bf_lab_time(fn, flush, repeats=16):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        flush.sum()
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b)*1000)
    graph.reset()
    del output
    return statistics.median(times)


def lab(deadline, weights):
    lines = [f'LAB_BF16 torch={torch.__version__} sm={torch.cuda.get_device_capability()}']
    # Proj has no expert bank and its router is unaligned. Reconstruct the public
    # layer weight distribution and use the exact moe_tune.py fixture on it.
    w = _bf__lab_make_weights(5353)
    native = _bf_v9_v9_prepare({**w, 'post_norm_w':w['post_expert_norm']})
    w13, w2 = _bf_expand(w['w13'],w['w13_sf']), _bf_expand(w['w2'],w['w2_sf'])
    lines.append(f'LAB_BF16 weights_bytes={w13.nbytes+w2.nbytes} fp32_global_epilogue=True')
    flush = torch.empty(128<<20, device='cuda', dtype=torch.uint8)
    for m in (128, 256):
        data = _bf_make_fixture(w,m,5353+m)
        ids, x = data['topk_ids'], data['hidden_states']
        counts = torch.bincount(ids[ids>=0], minlength=32).cpu().tolist()
        lines.append(f'LAB_ROUTE M={m} counts={counts} pairs={sum(counts)}')
        v9_us = _bf_lab_time(lambda:_bf_v9_v9_kernel_fn(data,native),flush)
        lines.append(f'LAB_V9 M={m} full_us={v9_us:.3f}')
        for threshold in (16, 32, 64, 96):
            if time.monotonic()>deadline-12:
                lines.append('LAB_BUDGET');return lines
            selected = [e for e,c in enumerate(counts) if c>=threshold]
            if not selected:
                continue
            # GPU grouped GEMM can skip zero-sized experts. Its backing tensor
            # keeps the maximum route capacity; only offs defines active rows.
            rounded = [((c+15)//16)*16 if c>=threshold else 0 for c in counts]
            offs = torch.tensor(rounded,device='cuda',dtype=torch.int32).cumsum(0).to(torch.int32)
            capacity = m*8+32*16
            packed_x = torch.zeros((capacity,4096), device='cuda',dtype=torch.bfloat16)
            expert_id = torch.zeros(capacity,device='cuda',dtype=torch.int64)
            start = 0
            for e in range(32):
                if rounded[e]:
                    pair = (ids==e).nonzero()
                    packed_x[start:start+counts[e]] = x[pair[:,0]]
                    expert_id[start:start+rounded[e]] = e
                    start += rounded[e]
            dense_w13, dense_w2 = w13.transpose(1,2), w2.transpose(1,2)
            def activate(y, idx):
                y=y.float()*w['w13_gs'][idx].reshape(-1,1)
                gate,up=y.chunk(2,-1)
                return (gate*torch.sigmoid(gate)*up).to(torch.bfloat16)
            cold=dict(data)
            hot=torch.tensor([c>=threshold for c in counts],device='cuda')
            cold['topk_ids']=torch.where((ids>=0)&hot[ids.clamp_min(0).long()],-1,ids)
            cold_native={**native,'grid':max(1,native['grid']//2)}
            cold_us=_bf_lab_time(lambda:_bf_v9_v9_kernel_fn(cold,cold_native),flush)
            common=f'M={m} threshold={threshold} experts={len(selected)} hot_rows={sum(counts[e] for e in selected)} padded={start} cold_us={cold_us:.3f}'
            for prefer_lt in (True, False):
                old_preference=torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm
                torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm=prefer_lt
                try:
                    def grouped():
                        y=torch._grouped_mm(packed_x,dense_w13,offs=offs)
                        z=activate(y,expert_id)
                        return torch._grouped_mm(z,dense_w2,offs=offs)
                    y=torch._grouped_mm(packed_x,dense_w13,offs=offs)
                    expected=packed_x[:counts[selected[0]]].float() @ w13[selected[0]].float().T
                    relative=float((y[:counts[selected[0]]].float()-expected).norm()/expected.norm())
                    us=_bf_lab_time(grouped,flush)
                    lines.append(f'LAB_GROUPED {common} lt={prefer_lt} dense_us={us:.3f} w13_l2={relative:.6f}')
                except Exception as exc:
                    lines.append(f'LAB_GROUPED_ERROR {common} lt={prefer_lt} {type(exc).__name__}: {str(exc)[:125]}')
                finally:
                    torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm=old_preference
            # Fixed-M vendor batched GEMM: executable with GPU routing and a
            # prepare-selected hot expert set, without reading counts on host.
            batch_x=torch.zeros((len(selected),m,4096),device='cuda',dtype=torch.bfloat16)
            for j,e in enumerate(selected):
                pair=(ids==e).nonzero();batch_x[j,:len(pair)] = x[pair[:,0]]
            bw13=w13[selected].transpose(1,2).contiguous()
            bw2=w2[selected].transpose(1,2).contiguous()
            bi=torch.tensor(selected,device='cuda')
            for out_dtype in (torch.bfloat16,torch.float32):
                def batched():
                    y=torch.bmm(batch_x,bw13,out_dtype=out_dtype).float()*w['w13_gs'][bi,None,None]
                    gate,up=y.chunk(2,-1)
                    z=(gate*torch.sigmoid(gate)*up).to(torch.bfloat16)
                    return torch.bmm(z,bw2,out_dtype=out_dtype)
                try:
                    us=_bf_lab_time(batched,flush)
                    lines.append(f'LAB_BATCHED {common} dense_us={us:.3f} output={out_dtype}')
                except Exception as exc:
                    lines.append(f'LAB_BATCHED_ERROR {common} {type(exc).__name__}: {str(exc)[:180]}')
            bw13=w13[selected].transpose(1,2)
            bw2=w2[selected].transpose(1,2)
            try:
                us=_bf_lab_time(batched,flush)
                lines.append(f'LAB_BATCHED_T {common} dense_us={us:.3f} output={out_dtype}')
            except Exception as exc:
                lines.append(f'LAB_BATCHED_T_ERROR {common} {type(exc).__name__}: {str(exc)[:125]}')
            del packed_x,batch_x,bw13,bw2
    return lines
