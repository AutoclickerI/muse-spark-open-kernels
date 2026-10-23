"""Assemble cand_wg2.py = cand_skinny.py + the wg generator (dev/wg_src.py) + configs."""
base = open("cand_skinny.py").read()
gen = open("dev/wg_src.py").read()
marker = "# (NT, TR, U, NA, XP, LD) instantiated"
wg_inst = '''
# (BT, NSUB, NCW, ST, MC): BT tokens per CTA, 64*NSUB*NCW weight rows, NCW consumer warpgroups,
# ST stages, MC CTAs sharing (multicasting) one activation tile.
WG_INSTANCES = [
    (8, 1, 1, 8, 1), (8, 1, 2, 6, 1), (16, 1, 1, 8, 1), (16, 1, 2, 6, 1), (32, 1, 1, 8, 1), (32, 1, 2, 6, 1),
    (64, 1, 1, 6, 1), (64, 1, 2, 4, 1), (128, 1, 1, 4, 1), (128, 1, 2, 4, 1), (128, 2, 2, 4, 1),
    (256, 1, 1, 4, 1), (256, 1, 2, 4, 1), (128, 2, 2, 4, 2), (256, 1, 2, 4, 2), (256, 1, 1, 4, 2),
]

'''
base = base.replace(marker, "# ---- Hopper TMA + wgmma kernel (sm_90a builds only) ----\n" + gen.split("\n", 1)[1] + wg_inst + marker, 1)
def rep(old, new):
    global base
    assert old in base, old[:80]
    base = base.replace(old, new)
rep('''def _source(instances) -> str:
    cases = " ".join(f"MP_CASE({nt}, {tr}, {u}, {na}, {xp}, {ld})" for nt, tr, u, na, xp, ld in instances)
    return CUDA_SRC.replace("MP_INSTANCES", cases)''', '''def _source(instances) -> str:
    cases = " ".join(f"MP_CASE({nt}, {tr}, {u}, {na}, {xp}, {ld})" for nt, tr, u, na, xp, ld in instances)
    return CUDA_SRC.replace("MP_INSTANCES", cases) + wg_source(WG_INSTANCES)''')
rep('''CPP_SRC = """
void skinny(''', '''CPP_SRC = """
void wg_gemm(torch::Tensor W, torch::Tensor X, torch::Tensor Out, int64_t N, int64_t bt, int64_t nsub,
             int64_t ncw, int64_t st, int64_t ks, int64_t mc);
void skinny(''')
rep('functions=["skinny"]', 'functions=["skinny", "wg_gemm"]')
rep('''        tag = hashlib.sha1(src.encode()).hexdigest()[:10]''', '''        arch = os.environ.get("MUSE_PROJ_ARCH") or ("9.0a" if _hopper() else None)
        tag = hashlib.sha1((src + str(arch)).encode()).hexdigest()[:10]
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch''')
rep('''            verbose=False,
        )
    return _MOD''', '''            verbose=False,
        )
        if arch:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _MOD


def _hopper() -> bool:
    return torch.cuda.get_device_capability() == (9, 0)''')
a = base.index("CONFIGS = {")
b = base.index("}\n", a) + 2
base = base[:a] + '''CONFIGS = {  # best on the H200 sweep of check 03113fd349d347dc
    ("qkv_gate", 1): (1, 4, 4, 1, 8, 1),
    ("qkv_gate", 2): (1, 4, 4, 1, 8, 1),
    ("qkv_gate", 4): (1, 2, 4, 1, 8, 1),
    ("qkv_gate", 8): (1, 2, 2, 4, 1, 1),
    ("o", 1): (1, 4, 1, 4, 1, 1),
    ("o", 2): (1, 4, 2, 4, 1, 0),
    ("o", 4): (1, 4, 2, 2, 1, 1),
    ("o", 8): (1, 2, 2, 2, 1, 1),
    ("pooler", 1): (1, 4, 1, 4, 1, 1),
    ("pooler", 2): (2, 4, 1, 8, 1, 1),
    ("pooler", 4): (1, 4, 1, 4, 1, 1),
    ("pooler", 8): (1, 2, 1, 4, 1, 1),
    ("unpooler", 1): (1, 4, 1, 8, 1, 1),
    ("unpooler", 2): (2, 4, 1, 4, 1, 1),
    ("unpooler", 4): (1, 4, 2, 2, 1, 1),
    ("unpooler", 8): (1, 2, 2, 2, 1, 1),
    ("router", 1): (1, 4, 2, 4, 8, 1),
    ("router", 2): (1, 4, 1, 8, 4, 1),
    ("router", 4): (1, 4, 1, 8, 4, 0),
    ("router", 8): (1, 2, 1, 8, 4, 0),
}
''' + base[b:]
rep('''def kernel_fn(data: Inputs, state) -> torch.Tensor:''', '''def _bt(m: int) -> int:
    return 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128 if m <= 128 else 256


# (layer, bt) -> (nsub, ncw, st, ks, mc): 64*nsub*ncw weight rows per CTA, K split over ks CTAs.
WG_CONFIGS = {
    ("qkv_gate", 8): (1, 1, 8, 4, 1), ("o", 8): (1, 1, 8, 1, 1), ("pooler", 8): (1, 1, 8, 2, 1), ("unpooler", 8): (1, 1, 8, 8, 1),
    ("qkv_gate", 16): (1, 1, 8, 4, 1), ("o", 16): (1, 1, 8, 1, 1), ("pooler", 16): (1, 1, 8, 2, 1), ("unpooler", 16): (1, 1, 8, 1, 1),
    ("qkv_gate", 32): (1, 1, 8, 4, 1), ("o", 32): (1, 1, 8, 1, 1), ("pooler", 32): (1, 1, 8, 2, 1), ("unpooler", 32): (1, 1, 8, 1, 1),
    ("qkv_gate", 64): (1, 1, 6, 4, 1), ("o", 64): (1, 1, 6, 1, 1), ("pooler", 64): (1, 1, 6, 2, 1), ("unpooler", 64): (1, 1, 6, 1, 1),
    ("qkv_gate", 128): (1, 2, 4, 4, 1), ("o", 128): (2, 2, 4, 4, 1), ("pooler", 128): (2, 2, 4, 8, 1), ("unpooler", 128): (2, 2, 4, 4, 1),
    ("qkv_gate", 256): (1, 2, 4, 4, 1), ("o", 256): (1, 2, 4, 2, 1), ("pooler", 256): (1, 2, 4, 4, 1), ("unpooler", 256): (1, 2, 4, 2, 1),
}
WG_MAX_M = 256


def wg_gemm(x: torch.Tensor, w: torch.Tensor, n: int, layer: str, cfg=None) -> torch.Tensor:
    """out[m, n] = x @ w[:n].T on Hopper (TMA + wgmma), BF16 in/out, FP32 accumulation."""
    m = x.shape[0]
    bt = _bt(m)
    nsub, ncw, st, ks, mc = cfg or WG_CONFIGS[(layer, bt)]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _module().wg_gemm(w, x, out, n, bt, nsub, ncw, st, ks, mc)
    return out


def kernel_fn(data: Inputs, state) -> torch.Tensor:''')
rep('''    if m <= SKINNY_MAX_M:
        return skinny_gemm(x, state[layer], n, layer)
    return F.linear(x, state[layer][:n])''', '''    if m <= WG_MAX_M and state.get("wg_ok"):
        return wg_gemm(x, state[layer], n, layer)
    if m <= SKINNY_MAX_M:
        return skinny_gemm(x, state[layer], n, layer)
    return F.linear(x, state[layer][:n])''')
open("cand_wg2.py", "w").write(base)
print("wrote cand_wg2.py")
