def v18s_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128)
    window = hist.unfold(0,8,1).sum(1)
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = v18s_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def dd_decodediet_tt_mw_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128)
    window = hist.unfold(0,8,1).sum(1)
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = dd_decodediet_tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def dd_decodediet_tt_mw_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 256), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = dd_decodediet_tt_mw_sm_c12_bytes(v)
        if not torch.equal(dd_decodediet_tt_mw_sm_c12_unpack(data, b, entries), v):
            return None
        table = dd_decodediet_tt_mw_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 256 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = dd_decodediet_tt_mw__to_i32(entry)
    else:
        data = dd_decodediet_tt_mw_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)