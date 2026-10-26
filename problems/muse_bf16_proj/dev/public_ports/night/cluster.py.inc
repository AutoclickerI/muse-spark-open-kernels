# Adapted from reviewed MIT operatorstack kernel; full notice in parent port_body.py.inc.
@triton.jit
def _ng_cluster_pool(x_desc, w_desc, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, SLOT_KB: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    pid_k = tl.program_id(0)  # rank in the cluster
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)
    if SLOT_KB == 8:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[8192]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    else:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[16384]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    started = tl.inline_asm_elementwise("barrier.cluster.arrive.relaxed.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                        dtype=tl.int32, is_pure=False, pack=1)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    # Peel the first weight tile. The opaque zero returned after GDC makes all
    # activation descriptor coordinates depend on the wait, including pipelined
    # future iterations. A source-only wait inside the loop permits TMA hoisting.
    wt0 = w_desc.load([(pid_n * kb_total + kb0) * BLOCK_N, 0])
    first = tl.gather(tl.reshape(wt0.to(tl.uint16,bitcast=True),(BLOCK_N*BLOCK_K,)),tl.full((1,),0,tl.int32),axis=0)
    witness = tl.sum(first.to(tl.int32),0)
    ready = tl.inline_asm_elementwise("griddepcontrol.wait; and.b32 $0, $1, 0;", "=r,r", [witness], dtype=tl.int32, is_pure=False, pack=1)
    xm = tl.multiple_of(m0 + ready, BLOCK_M)
    xt0 = x_desc.load([xm, k]).T
    acc = tl.dot(wt0, xt0, tl.zeros((BLOCK_N, BLOCK_M),dtype=tl.float32))
    k += BLOCK_K
    for _ in range(kb0+1, kb1):
        wt = w_desc.load([(pid_n * kb_total + k // BLOCK_K) * BLOCK_N, 0])
        xt = x_desc.load([xm, k]).T
        acc = tl.dot(wt, xt, acc)
        k += BLOCK_K
    idx4 = (tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]) * 4
    TILE_BYTES: tl.constexpr = BLOCK_N * BLOCK_M * 4
    if pid_k != 0:
        tl.inline_asm_elementwise("st.shared.f32 [$1], $2; mov.u32 $0, 0;", "=r,r,f", [base + idx4, acc],
                                  dtype=tl.int32, is_pure=False, pack=1)
        tl.debug_barrier()
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        dst = base + (pid_k - 1) * TILE_BYTES + ready
        if TILE_BYTES == 4096:
            tl.inline_asm_elementwise(CBF_V4COPY2, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        else:
            tl.inline_asm_elementwise(CBF_V4COPY4, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                  dtype=tl.int32, is_pure=False, pack=1)
    else:
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        done = tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; "
                                         "barrier.cluster.wait.acquire.aligned; mov.u32 $0, 0;", "=r,r",
                                         [pid_k + ready], dtype=tl.int32, is_pure=False, pack=1)
        tot = acc
        for s in tl.static_range(1, SPLIT_K):
            tot += tl.inline_asm_elementwise("ld.shared.f32 $0, [$1];", "=f,r",
                                             [base + (s - 1) * TILE_BYTES + done + idx4], dtype=tl.float32,
                                             is_pure=False, pack=1)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
        tl.store(out_ptrs, tot, mask=offs_m[None, :] < M)
