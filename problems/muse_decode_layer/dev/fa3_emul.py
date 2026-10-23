"""Run a candidate's FA3 path locally with flash_attn_with_kvcache emulated in torch."""
import sys, types, importlib.util, math
import torch
repo = "/home/jihoon/competitions/muse_kernel"
pdir = repo + "/problems/muse_decode_layer"
sys.path[:0] = [pdir]
cand = sys.argv[1]

def emul(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale,
         causal, window_size, scheduler_metadata=None, k_descale=None, v_descale=None):
    T, H, D = q.shape
    B = cache_seqlens.shape[0]
    L = max_seqlen_q
    page = k_cache.shape[1]
    kf = k_cache.view(-1, 2, 64).float(); vf = v_cache.view(-1, 2, 64).float()
    if k_descale is not None:
        kf = kf * k_descale[0, 0]; vf = vf * v_descale[0, 0]
    out = torch.empty((T, H, D), device=q.device, dtype=torch.bfloat16)
    qf = q.float()
    for b in range(B):
        n = int(cache_seqlens[b]); seq = n - L
        pos = torch.arange(n, device=q.device)
        rows = page_table[b].long()[pos // page] * page + pos % page
        K = kf[rows].repeat_interleave(8, 1); V = vf[rows].repeat_interleave(8, 1)  # [n, 16, 64]
        for j in range(L):
            p = seq + j
            vis = pos <= p
            if window_size[0] >= 0:
                vis &= pos >= p - window_size[0]
            s = torch.einsum("hd,nhd->hn", qf[b * L + j], K) * softmax_scale
            s = s.masked_fill(~vis[None], float("-inf")).softmax(-1)
            out[b * L + j] = torch.einsum("hn,nhd->hd", s, V).to(torch.bfloat16)
    return out

mod = types.ModuleType("sgl_kernel.flash_attn"); mod.flash_attn_with_kvcache = emul
sys.modules["sgl_kernel.flash_attn"] = mod
spec = importlib.util.spec_from_file_location("cand", cand)
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
c.ATTN_BACKEND = "fa3"
spec = importlib.util.spec_from_file_location("reference", pdir + "/reference.py")
ref = importlib.util.module_from_spec(spec); sys.modules["reference"] = ref; spec.loader.exec_module(ref)
ref._fa3_scheduler_metadata = lambda *a, **k: None
w = ref.generate_weights(ref.WEIGHTS_SEED)
state = c.prepare({k: v.clone() for k, v in w.items()})
cases = [dict(batch=1, ctx=10, layer="sliding"), dict(batch=3, ctx=6000, layer="full", q_len=8, dist="outliers"),
         dict(batch=4, ctx=7000, layer="sliding", q_len=8, page=1, kv_scale=0.015625),
         dict(batch=4, ctx=3000, layer="sliding", page=1, spread="mixed"), dict(batch=16, ctx=5000, layer="full", page=1)]
for i, case in enumerate(cases):
    case = {**case, "seed": 1234 + i}
    case.update(ref.case_fields(case, w))
    data = ref.generate_input(**case)
    out = c.kernel_fn(data, state)
    print(case, ref.check_implementation(data, ref.collect(data, out), w))
