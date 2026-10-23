"""Freeze the canonical RS128 grouped reader into a complete prefill candidate."""
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_canonical import frozen_python, separate_global_scales

root = HERE.parent
base = (root / "cand_canonical_prefill.py").read_text()
front = base[:base.index("\nCUDA_SRC = ")]
front = re.sub(r'\A(# okc:[^\n]*\n)""".*?"""',
               r'\1"""Prefill with one canonical V6 expert bank and adaptive RS128 WGMMA groups."""',
               front, flags=re.S)
front = front.replace("import math\n", "import math\nimport os\n")
front = front.replace("    if False:\n        _bench_w4a16(weights)\n", "")
if "    if T >= BIG_TOKENS or st[\"bank\"] is None:" in front:
    start = front.index("    if T >= BIG_TOKENS or st[\"bank\"] is None:")
    end = front.index("    if T <= 8:", start)
    front = front[:start] + "    moe = experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st)\n" + front[end:]
# These raw-bank readers have no call sites in the one-bank candidate.
for name in ("_marlin_bank", "_bench_w4a16", "_experts_marlin", "_experts_dequant"):
    front = re.sub(r"\ndef " + name + r"\(.*?(?=\ndef |\Z)", "\n", front, flags=re.S)

source = separate_global_scales(frozen_python("rs128").GROUPED_SRC)
a = source.index("__global__ void histogram(")
b = source.index("template<int N,int MODE> __device__", a)
source = source[:a] + source[b:]
a = source.index("void route(torch::Tensor")
b = source.index("void gemm(torch::Tensor", a)
source = source[:a] + source[b:]
(HERE / "canonical_grouped_rs128_separate.cu").write_text(source)

packer = base[base.index("_DQ_SLOTS = "):base.index("\ndef prepare(weights):", base.index("_DQ_SLOTS = "))]
prepare = base[base.index("\ndef prepare(weights):", base.index("_DQ_SLOTS = ")):]
prepare = prepare.replace("st['tiled_mod']=load_module(None)", "st['grouped_mod']=_grouped_module()")
host = '''
_mod = None

def _grouped_module():
    global _mod
    if _mod is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        cap = torch.cuda.get_device_capability()
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a" if cap == (9, 0) else f"{cap[0]}.{cap[1]}"
        try:
            _mod = load_inline(
                name="muse_prefill_canonical_grouped_" + hashlib.sha256(GROUPED_SRC.encode()).hexdigest()[:14],
                cpp_sources="void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);",
                cuda_sources=GROUPED_SRC, functions=["gemm"],
                extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"], verbose=False)
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous
    return _mod


@triton.jit
def _canonical_plan(COUNTS, OFFSETS, TASKS):
    e = tl.arange(0, 32)
    c = tl.load(COUNTS + e)
    inc = tl.cumsum(c, 0)
    tl.store(OFFSETS + e, inc - c)
    tl.store(OFFSETS + 32, tl.sum(c, 0))
    chosen = tl.where(c <= 8, 8, tl.where(c <= 16, 16, tl.where(c <= 32, 32, tl.where(c <= 64, 64, 128))))
    for i in tl.static_range(5):
        width = 8 << i
        tasks = tl.where(chosen == width, (c + width - 1) // width, 0)
        inclusive = tl.cumsum(tasks, 0)
        tl.store(TASKS + i * 33 + e, inclusive - tasks)
        tl.store(TASKS + i * 33 + 32, tl.sum(tasks, 0))


@triton.jit
def _canonical_pairs(SLOT_E, SLOT_POS, OFFSETS, PAIRS, POSITIONS, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(SLOT_E + j, j < SIZE, -1)
    local = e >= 0
    off = tl.load(OFFSETS + e, local, 0)
    pos = off + tl.load(SLOT_POS + j, j < SIZE, 0)
    tl.store(PAIRS + pos, j, (j < SIZE) & local)
    tl.store(POSITIONS + j, tl.where(local, pos, -1), j < SIZE)


@triton.jit
def _canonical_finalize(Y, POS, PROBS, NORM, OUT, D: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, D)
    norm = tl.load(NORM + c).to(tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    for k in range(8):
        pos = tl.load(POS + t * 8 + k)
        y = tl.load(Y + pos.to(tl.int64) * D + c, pos >= 0, 0).to(tl.float32) * norm
        y *= tl.rsqrt(tl.sum(y * y, 0) / D + 1e-8)
        acc += tl.load(PROBS + t * 8 + k) * y
    tl.store(OUT + t * D + c, acc.to(tl.bfloat16))


def run_grouped_raw(xn, topk_w, slot_e, slot_pos, counts, st):
    T, D = xn.shape
    dev = xn.device
    offsets = torch.empty(33, dtype=torch.int32, device=dev)
    tasks = torch.empty(5 * 33, dtype=torch.int32, device=dev)
    pairs = torch.empty(T * 8, dtype=torch.int32, device=dev)
    positions = torch.empty(T * 8, dtype=torch.int32, device=dev)
    _canonical_plan[(1,)](counts, offsets, tasks, num_warps=1)
    _canonical_pairs[(triton.cdiv(T * 8, 256),)](slot_e, slot_pos, offsets, pairs, positions, SIZE=T * 8, BLOCK=256)
    inter = torch.empty((T * 8, D), dtype=torch.bfloat16, device=dev)
    y = torch.empty_like(inter)
    mod = st['grouped_mod']
    mod.gemm(st['w13p'], st['gs13_gate'], st['gs13_up'], xn, inter, pairs, offsets, tasks, 1, 0, st['sms'] * 4, 2)
    mod.gemm(st['w2p'], st['gs2'], st['gs2'], inter, y, pairs, offsets, tasks, 0, 0, st['sms'] * 4, 2)
    out = torch.empty_like(xn)
    _canonical_finalize[(T,)](y, positions, topk_w, st['post_expert'], out, D=D, num_warps=8)
    return out


def experts_grouped(pooled, topk_w, slot_e, slot_pos, counts, st):
    xn = torch.empty_like(pooled)
    _pre_expert_norm_kernel[(pooled.shape[0],)](pooled, st['pre_expert'], xn, D=D_EXPERT, num_warps=4)
    return run_grouped_raw(xn, topk_w, slot_e, slot_pos, counts, st)

'''
front = front.replace("moe = experts_tiled(", "moe = experts_grouped(")
out = front + "\nGROUPED_SRC = " + repr(source) + "\n" + host + packer + prepare
(root / "cand_canonical_grouped.py").write_text(out)
print(root / "cand_canonical_grouped.py")
