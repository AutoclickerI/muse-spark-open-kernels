"""Local arithmetic + fresh graph checks; explicit stand-in for Hopper DG ABI."""
import importlib.util
import json
from pathlib import Path
import time
import torch

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD = ROOT / 'prof/lab/moe_decode_adapters.py'
spec = importlib.util.spec_from_file_location('dc_screen_payload', PAYLOAD)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
started = time.monotonic()
report = []
for i, seed in enumerate(p.DC_SEEDS):
    w = p.dc_weights(seed, 'cuda')
    native, st = p.dc_prepare(w, local_glue=True)
    if i == 0:
        # Stand-in tests ABI only; separately verify the real per-call DG
        # expansion against an independent raw E2M1/E4M3 interpretation.
        counts = torch.zeros(32, device='cuda', dtype=torch.int32)
        counts[0] = 1
        scratch = torch.empty((32, 8192, 4096), device='cuda', dtype=torch.bfloat16)
        lut = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device='cuda')
        for key, gated, nr in (('w13', True, 8192), ('w2', False, 4096)):
            target = scratch.view(-1)[:32 * nr * 4096].view(32, nr, 4096)
            bank = st['w13p' if gated else 'w2p']
            p.pf__canonical_expand_coalesced[(32, nr // 16, 4)](bank, bank.view(torch.int32), target.view(torch.int32), counts, ROWS=nr, GATED=gated, KB=1024, num_warps=8)
            q = w[key][0]
            c = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
            expected = (lut[c & 7] * torch.where(c & 8 != 0, -1., 1.) * w[key+'_sf'][0].float().repeat_interleave(16, -1)).bfloat16()
            assert torch.equal(target[0], expected), (key, 'DG per-call expansion mismatch')
        del counts, scratch, target, bank, q, c, expected, lut
        print('DG canonical W13/W2 raw-scale expansion BIT_EXACT', flush=True)
    for m in (128, 256):
        for r0 in ((True, False) if i == 0 else (False,)):
            ds = p.dc_fixtures(w, m, r0=r0)
            wants = p.dc_oracles(ds, w)
            for name in p.DC_NAMES:
                g = p.MWGraph(lambda d, n=name: p.dc_call(d, native, st, n), ds[0])
                if name == 'dg_w8':
                    for d in ds:
                        g.refresh(d)
                        got = g.replay()
                        torch.cuda.synchronize()
                        assert got.shape == (m, 4096) and got.dtype == torch.bfloat16
                        assert bool((got == 0).all())
                    report.append(dict(seed=seed, m=m, r0=r0, name=name, abi=True, arithmetic=False))
                else:
                    err, row = p.dc_gate(g, ds, wants)
                    assert err <= .01 and row <= .05, (seed, m, r0, name, err, row)
                    report.append(dict(seed=seed, m=m, r0=r0, name=name, l2=err, row=row))
                del g
    print(f'SCREEN seed={seed} PASS rows={len(report)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    del st, native, w, ds, wants
    torch.cuda.empty_cache()
out = ROOT / 'runs/screen/moe_decode_adapters.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(dict(rows=report, elapsed=time.monotonic()-started), indent=2) + '\n')
print(f'PASS {len(report)} rows: {sum("l2" in r for r in report)} real arithmetic; {sum(r.get("abi",False) for r in report)} DG stand-in ABI only. {out}', flush=True)
