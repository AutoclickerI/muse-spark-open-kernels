def _serving_dense(state, x, scores, ids):
    """V9 at verify256; bounded V9 chunks above its shared-route capacity.

    Dense slot maps stay global to the original call, including remote -1s.
    Only shape/device metadata chooses chunk sizes; no host data dependency.
    """

    mod = state["mod"]
    # Route positions are int16. Each row needs 8 expert slots and 40 shared
    # bytes; use the actual compiled kernel's smem formula, capped below32768.
    # Large routing tables substantially slow the persistent reader. Bound
    # prefill to512 rows as well as the hardware/int16 limit; verify<=256 is
    # unchanged. No synthetic route timing selects a different producer.
    lo, hi = 1, 512
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mod.smem_needed(1, 1, mid) <= mod.smem_optin:
            lo = mid
        else:
            hi = mid - 1
    capacity = max(1, lo // 8 * 8)
    rows, maps = [], []
    for begin in range(0, x.shape[0], capacity):
        chunk = x[begin : begin + capacity]
        data = {
            "hidden_states": chunk,
            "topk_ids": ids[begin : begin + capacity],
            "topk_weights": scores[begin : begin + capacity],
            "config": {
                "local_count": NLOC,
                "d_hidden": H,
                "d_expert": H,
                "local_start": state["local_start"],
            },
        }
        dense, mapping = _serving_v9_call(data, state, return_dense=True)
        rows.append(dense)
        maps.append(
            torch.where(mapping >= 0, mapping + begin * TOPK, mapping)
            if begin
            else mapping
        )
    return (rows[0], maps[0]) if len(rows) == 1 else (torch.cat(rows), torch.cat(maps))
