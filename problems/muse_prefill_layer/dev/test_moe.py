import sys, time, torch
sys.path.insert(0, "problems/muse_prefill_layer"); sys.path.insert(0, "problems/muse_prefill_layer/dev")
from nvfp4 import random_weights, dequantize
import moe_host as H

torch.manual_seed(0)
dev = "cuda"
E, D = 32, 4096
gen = torch.Generator(device=dev).manual_seed(1)
w13, w13_sf, gs13 = random_weights(2 * D, D, generator=gen, device=dev, leading=(E,))
w2, w2_sf, gs2 = random_weights(D, D, generator=gen, device=dev, leading=(E,))
gs13 = (gs13 * torch.exp(torch.randn(E, generator=gen, device=dev))).float()
gs2 = (gs2 * torch.exp(torch.randn(E, generator=gen, device=dev))).float()
t0 = time.time()
b13, s13 = H.repack(w13, w13_sf, gated=True)
b2, s2 = H.repack(w2, w2_sf, gated=False)
print("repack", time.time() - t0, b13.shape, s13.shape)
t0 = time.time()
mod = H.load_module("problems/muse_prefill_layer/dev/moe_w4a16.cu")
print("compile", time.time() - t0)

T = int(sys.argv[1]) if len(sys.argv) > 1 else 300
# routing: each token picks up to 8 local experts (skewed)
probs = torch.softmax(torch.randn(E, device=dev) * 1.5, 0)
slot_e = torch.full((T, 8), -1, dtype=torch.int32, device=dev)
for t in range(T):
    n = int(torch.randint(0, 4, (1,)))
    es = torch.multinomial(probs, n, replacement=False) if n else torch.empty(0, dtype=torch.long, device=dev)
    slot_e[t, :n] = es.int()
counts = torch.zeros(E, dtype=torch.int32, device=dev)
slot_pos = torch.zeros((T, 8), dtype=torch.int32, device=dev)
for t in range(T):
    for k in range(8):
        e = int(slot_e[t, k])
        if e >= 0:
            slot_pos[t, k] = counts[e]; counts[e] += 1
print("counts", counts.tolist())
x = (torch.randn(T, D, device=dev)).to(torch.bfloat16)
tstart = torch.zeros(33, dtype=torch.int32, device=dev)
H.tile_offsets_kernel[(1,)](counts, tstart, BNC=H.BN)
max_tiles = (T * 8 + E * (H.BN - 1)) // H.BN + 1
xt = torch.full((max_tiles * H.BN * D,), float("nan"), dtype=torch.bfloat16, device=dev)
rows = torch.empty((T, 8), dtype=torch.int32, device=dev)
H.gather_tiled_kernel[(T,)](x, slot_e, slot_pos, tstart, xt, rows, D=D, BNC=H.BN)
inter = torch.empty((max_tiles * H.BN * D,), dtype=torch.bfloat16, device=dev)
y = torch.empty((max_tiles * H.BN * D,), dtype=torch.bfloat16, device=dev)
nsm = torch.cuda.get_device_properties(0).multi_processor_count
t0 = time.time()
mod.moe_gemm(b13, s13, gs13, xt, inter, tstart, 8192 // 256, D // 64, D // 64, 1, nsm)
mod.moe_gemm(b2, s2, gs2, inter, y, tstart, 4096 // 256, D // 64, D // 64, 0, nsm)
torch.cuda.synchronize()
print("gemm", time.time() - t0)
# reference per route
valid = rows >= 0
rr = rows[valid].long()
toks = torch.arange(T, device=dev)[:, None].expand(T, 8)[valid]
es = slot_e[valid].long()
yi = torch.empty((rr.numel(), D), dtype=torch.bfloat16, device=dev)
H.untile_rows_kernel[(rr.numel(),)](y, rr.int().contiguous(), yi, D=D, BNC=H.BN)
ii = torch.empty((rr.numel(), D), dtype=torch.bfloat16, device=dev)
H.untile_rows_kernel[(rr.numel(),)](inter, rr.int().contiguous(), ii, D=D, BNC=H.BN)
errs, errs1 = [], []
for e in range(E):
    sel = es == e
    if not sel.any():
        continue
    xe = x[toks[sel]].float()
    W1 = dequantize(w13[e], w13_sf[e], gs13[e])
    h = xe @ W1.T
    it = (torch.nn.functional.silu(h[:, :D]) * h[:, D:])
    W2 = dequantize(w2[e], w2_sf[e], gs2[e])
    ref = it.to(torch.bfloat16).float() @ W2.T
    errs1.append(((ii[sel].float() - it).norm() / it.norm()).item())
    errs.append(((yi[sel].float() - ref).norm() / ref.norm()).item())
print("inter rel err max", max(errs1), "y rel err max", max(errs))
# timing
s, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for _ in range(2):
    mod.moe_gemm(b13, s13, gs13, xt, inter, tstart, 32, 64, 64, 1, nsm)
s.record()
for _ in range(5):
    mod.moe_gemm(b13, s13, gs13, xt, inter, tstart, 32, 64, 64, 1, nsm)
e_.record(); torch.cuda.synchronize()
us = s.elapsed_time(e_) / 5 * 1e3
ntiles = int(tstart[32])
print(f"gemm1 {us:.1f} us, tiles {ntiles}, {2 * ntiles * 128 * 8192 * 4096 / us / 1e6:.0f} TF (padded), {2 * int(counts.sum()) * 8192 * 4096 / us / 1e6:.0f} TF (real)")
