import torch, time, sys
sys.path.insert(0, "problems/muse_prefill_layer")
from nvfp4 import random_weights, dequantize
from quack.gemm_w4 import gemm_w4a16, prepare_w4_weight
gen = torch.Generator(device="cuda").manual_seed(0)
N, K = 8192, 4096
q, sf, gs = random_weights(N, K, generator=gen, device="cuda")
w = dequantize(q, sf, gs)  # fp32
t0=time.time()
blob, sfb = prepare_w4_weight(q, sf, "nvfp4")
print("prepare", time.time()-t0, blob.shape, blob.dtype, sfb.shape, sfb.dtype)
for M in (64, 256, 1024):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    t0=time.time()
    out = gemm_w4a16(x, blob, sfb, tensor_scale=float(gs), wformat="nvfp4")
    torch.cuda.synchronize()
    print("first call", M, time.time()-t0)
    ref = x.float() @ w.T
    err = ((out.float()-ref).norm()/ref.norm()).item()
    for _ in range(3): gemm_w4a16(x, blob, sfb, tensor_scale=float(gs), wformat="nvfp4")
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20): gemm_w4a16(x, blob, sfb, tensor_scale=float(gs), wformat="nvfp4")
    e.record(); torch.cuda.synchronize()
    us = s.elapsed_time(e)/20*1e3
    s.record()
    for _ in range(20): torch.mm(x, w.to(torch.bfloat16).T)
    e.record(); torch.cuda.synchronize()
    us2 = s.elapsed_time(e)/20*1e3
    print(f"M={M} err={err:.2e} w4a16 {us:.1f} us ({2*M*N*K/us/1e6:.1f} TF)  bf16 mm {us2:.1f} us")
