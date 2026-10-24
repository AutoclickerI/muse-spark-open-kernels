# Each decoded tensor below is diagnostic scratch; prepare returns only the host's original state.
def sm_unpacked_tile(tile):
    words = tile[:32768].contiguous().view(torch.int32).to(torch.int64).view(8, 8, 32, 2, 2)
    codes = torch.zeros((8, 8, 32, 2, 8, 2), device=tile.device, dtype=torch.int64)
    for k, h, bit, word, pos in _TABLE:
        codes[..., k, h] |= ((words[..., 0 if word == 'A' else 1] >> pos) & 1) << bit
    codes = codes.view(1, 8, 8, 8, 4, 2, 2, 2, 2, 2)
    codes = codes.permute(0, 8, 3, 1, 2, 4, 5, 6, 7, 9).reshape(16, 4096).byte()
    sf = tile[32768:].view(1, 8, 8, 4, 8, 2).permute(0, 5, 2, 1, 4, 3).reshape(16, 256)
    return codes, sf


def sm_bank_check(weights, bank):
    checked = 0
    for name, tiles in (('w13', (0, 255, 511)), ('w2', (0, 127, 255))):
        for expert in range(32):
            for tile in tiles:
                rows = torch.arange(8 if name == 'w13' else 16, device=bank['w13p'].device)
                rows = torch.cat((tile * 8 + rows, 4096 + tile * 8 + rows)) if name == 'w13' else tile * 16 + rows
                packed = weights[name][expert, rows]
                expected = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
                codes, scales = sm_unpacked_tile(bank[name + 'p'][expert, tile])
                for kind, got, want in (('codes', codes, expected), ('E4M3', scales, weights[name + '_sf'][expert].view(torch.uint8)[rows])):
                    mismatch = (got != want).nonzero()
                    if mismatch.numel():
                        r, k = mismatch[0].tolist()
                        return False, f'BANK_FAIL {name} {kind} expert={expert} row={int(rows[r])} k={k} got={int(got[r,k])} want={int(want[r,k])}'
                checked += 1
    return True, f'BANK_PASS tiles={checked} codes_and_E4M3_bytes=exact; same bank for NT1/NT2/NT4/v9m1/hotNT1'


def sm_run_scratch(data, bank, mod, nt, stages=2):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].int().contiguous()
    tw = data['topk_weights'].float().contiguous()
    m = x.shape[0]
    while stages > 1 and mod.smem_needed(nt, stages, m) > mod.smem_optin:
        stages -= 1
    if mod.smem_needed(nt, stages, m) > mod.smem_optin:
        raise ValueError('small-M shared memory exceeds capacity')
    inter = torch.full((m * 8, H), float('nan'), device=x.device, dtype=torch.bfloat16)
    y = torch.full((m * 8, H), float('nan'), device=x.device)
    ss = torch.full((m * 8,), float('nan'), device=x.device)
    out = torch.empty_like(x)
    counters = torch.zeros_like(bank['cnt'])
    mod.run(x, ids, tw, bank['pnw'], bank['w13p'], bank['w2p'], bank['gs13'], bank['gs2'],
            inter, y, ss, counters, out, data['config']['local_start'], nt, stages,
            bank['grid'] * mod.resident_blocks(nt, stages, m))
    torch.cuda.synchronize()
    return inter, y, ss, out


def sm_dequant(weights, name, expert):
    packed = weights[name][expert]
    code = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    grid = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=packed.device)
    values = grid[(code & 7).long()] * torch.where(code & 8 != 0, -1., 1.)
    return (values.view(-1, 256, 16) * weights[name + '_sf'][expert].float()[..., None] *
            weights[name + '_gs'][expert]).reshape(-1, 4096)


def sm_error(got, want):
    g, w = got.float(), want.float()
    delta = (g - w).norm(dim=-1)
    denom = w.norm(dim=-1)
    rel = float((g - w).norm() / w.norm().clamp_min(1e-20))
    row = float((delta / denom.clamp_min(.1 * denom.mean()).clamp_min(1e-20)).max())
    return rel, row


def sm_first_difference(stage, got, want, expert, tokens, tol=.015):
    g, w = got.float(), want.float()
    if g.ndim == 1:
        g, w = g[:, None], w[:, None]
    errors = (g - w).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-15)
    bad = ((errors > tol) | ~torch.isfinite(g).all(-1)).nonzero()
    if not bad.numel():
        return None
    row = int(bad[0, 0])
    col = int((g[row] - w[row]).abs().argmax())
    return (f'FIRST_FAIL stage={stage} expert={expert} token={int(tokens[row])} row={row} col={col} '
            f'got={float(g[row,col]):.7g} want={float(w[row,col]):.7g} row_rel={float(errors[row]):.6g}')


def sm_stage_check(data, weights, base, test, label):
    ids = data['topk_ids'] - data['config']['local_start']
    offset = 0
    result = torch.zeros_like(data['hidden_states'], dtype=torch.float32)
    first = None
    for expert in range(32):
        tokens, slots = (ids == expert).nonzero(as_tuple=True)
        count = tokens.numel()
        if not count:
            continue
        sl = slice(offset, offset + count)
        x = data['hidden_states'][tokens].float()
        h = x @ sm_dequant(weights, 'w13', expert).T
        activation = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = activation @ sm_dequant(weights, 'w2', expert).T
        weighted = y * (1. + weights['post_norm_w'].float())
        normalized = weighted * torch.rsqrt(weighted.square().mean(-1, keepdim=True) + 1e-8)
        result.index_add_(0, tokens, normalized * data['topk_weights'][tokens, slots, None])
        for stage, got, want in (
            ('control_swiglu', base[0][sl], activation),
            (label + '_swiglu', test[0][sl], activation),
            ('control_w2', base[1][sl], y),
            (label + '_w2', test[1][sl], y),
            (label + '_ss', test[2][sl], (test[1][sl] * (1. + weights['post_norm_w'].float())).square().sum(-1)),
        ):
            mismatch = sm_first_difference(stage, got, want, expert, tokens)
            if mismatch and first is None:
                first = mismatch
        offset += count
    rel, row = sm_error(test[3], result)
    base_rel, base_row = sm_error(base[3], result)
    status = 'PASS' if first is None and rel <= .01 and row <= .05 and base_rel <= .01 and base_row <= .05 else 'FAIL'
    return result, status, first, f'STAGE_{status} {label} fp32={rel:.6f}/{row:.6f} control={base_rel:.6f}/{base_row:.6f} pairs={offset}'


def sm_small_call(data, state, nt=1):
    return v9_small_nt1_kernel_fn(data, state) if nt == 1 else v9_nt4_kernel_fn(data, state)


def lab(deadline, weights):
    device = weights['w13'].device
    props = torch.cuda.get_device_properties(device)
    lines = [f'META small-M contract diagnostic {props.name} sm={props.major}{props.minor}; raw E4M3; independent FP32 stages; paired cold graphs',
             'CAUSE old lab fed prepare-recoded scales to raw-E4M3 NT4/v9m1; e.g. raw128=0x70 became0xe0=-32 for the raw reader']
    if lab_time.monotonic() > deadline - 20:
        return lines + ['BUDGET before_bank']
    w = dict(weights, post_norm_w=weights.get('post_norm_w', weights.get('post_expert_norm')))
    bank = v9_prepare(w)
    ok, bank_line = sm_bank_check(w, bank)
    lines.append(bank_line)
    if not ok:
        return lines
    mod = _small_nt4_module()
    small = dict(bank, mod=mod, cnt=torch.zeros_like(bank['cnt']))
    rw, source = lab_router(weights, device)
    lines[0] += ' router=' + source
    flush = torch.empty(128 << 20, device=device, dtype=torch.uint8)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    passed = 0
    try:
        for m, nt in ((1, 1), (8, 1), (16, 4), (32, 4)):
            if lab_time.monotonic() > deadline - 10:
                lines.append(f'BUDGET before_M={m}')
                break
            datasets = lab_datasets(rw, m, 'R', device)
            d = datasets[0]
            base_scratch = sm_run_scratch(d, bank, bank['mod'], 1)
            test_scratch = sm_run_scratch(d, bank, mod, nt)
            want, status, first, message = sm_stage_check(d, w, base_scratch, test_scratch, f'NT{nt}_M{m}')
            lines.append(message)
            if first:
                lines.append(first)
            if status != 'PASS':
                return lines + [f'REJECT M={m} before_timing']
            old = LabGraph(lambda d: v9_kernel_fn(d, bank), d)
            new = LabGraph(lambda d, nt=nt: sm_small_call(d, small, nt), d)
            for fresh in datasets:
                old.refresh(fresh)
                control = old.replay().clone()
                new.refresh(fresh)
                got = new.replay()
                torch.cuda.synchronize()
                rel, row = sm_error(got, control)
                if not bool(torch.isfinite(got).all()) or rel > .01 or row > .05:
                    a = sm_run_scratch(fresh, bank, bank['mod'], 1)
                    b = sm_run_scratch(fresh, bank, mod, nt)
                    _, _, first, message = sm_stage_check(fresh, w, a, b, f'refresh_NT{nt}_M{m}')
                    return lines + [message, first or f'FIRST_FAIL output M={m} rel={rel:.6g}/{row:.6g}']
            paired, median, anchor = lab_pair(old, new, datasets, flush, repeats=8)
            lines.append(f'T M={m} NT1={anchor:.2f} newNT{nt}={paired:.2f} rawmedian={median:.2f} ratio={paired/anchor:.4f}')
            passed += 1
            del old, new, base_scratch, test_scratch, want
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return lines + [f'SUMMARY completed={passed}/4 raw_bank_exact={ok} left_s={deadline-lab_time.monotonic():.1f}; timing is H200 evidence only on sm90']
