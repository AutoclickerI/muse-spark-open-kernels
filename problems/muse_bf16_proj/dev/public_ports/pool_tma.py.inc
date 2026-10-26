@triton.jit
def _pp_pool_tma_early(x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr):
    """_skinny_gemm with weight and activation tiles fetched by TMA (host-built tensor descriptors);
    rows of x beyond M are zero-filled by the copy engine. K is cut into SPLIT_K ranges of whole blocks."""
    tl.extra.cuda.gdc_launch_dependents()
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    wt0 = w_desc.load([(pid_n * kb_total + kb0) * BLOCK_N, 0])
    first = tl.gather(tl.reshape(wt0.to(tl.uint16,bitcast=True),(BLOCK_N*BLOCK_K,)),tl.full((1,),0,tl.int32),axis=0)
    witness = tl.sum(first.to(tl.int32),0)
    ready = tl.inline_asm_elementwise("griddepcontrol.wait; and.b32 $0, $1, 0;", "=r,r", [witness], dtype=tl.int32, is_pure=False, pack=1)
    xm = tl.multiple_of(m0 + ready, BLOCK_M)
    xt0 = x_desc.load([xm, k]).T
    acc = tl.dot(wt0, xt0, tl.zeros((BLOCK_N,BLOCK_M),dtype=tl.float32))
    if HILO:acc = tl.dot(w2_desc.load([n0,k]),xt0,acc)
    k += BLOCK_K
    for kb in range(kb0+1, kb1):
        wt = w_desc.load([(pid_n * kb_total + kb) * BLOCK_N, 0])
        xt = x_desc.load([xm, k]).T
        acc = tl.dot(wt, xt, acc)
        if HILO:
            acc = tl.dot(w2_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc, mask=m_mask)
    else:
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(1) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot, mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")
