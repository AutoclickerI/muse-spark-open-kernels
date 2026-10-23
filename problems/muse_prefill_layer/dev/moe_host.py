"""Host side of the grouped NVFP4 W4A16 expert GEMM (dev copy; inlined into candidates)."""
import os

import torch
import triton
import triton.language as tl

BN = 128          # tokens per tile (must match the CUDA kernel)
D = 4096          # MoE width (K of both GEMMs, output of GEMM2, intermediate of GEMM1)


def _blob_index(dev):
    """quack's per-m64 repack: thread t's 16 bytes per (m64, k64) tile -> (row in m64, byte in k-tile)."""
    t = torch.arange(128, device=dev)
    j = torch.arange(16, device=dev)
    b, p = j >> 2, j & 3
    r16 = 16 * (t >> 5) + (t & 31) // 4
    c = t & 3
    row = r16[:, None] + 8 * (p & 1)[None, :]                    # (128, 16)
    col = 8 * b[None, :] + c[:, None] + 4 * (p >> 1)[None, :]    # (128, 16) byte within the 32-byte k-tile row
    s = torch.arange(32, device=dev)
    srow = 16 * (s >> 3) + (s & 7)                               # (32,)
    return row, col, srow


def cta_row_order(m, gated, dev):
    """Weight row permutation into 256-row CTA blocks. Gated (w13: gate rows then up rows):
    block fb = [gate fb*128 + 0..63, up .., gate fb*128 + 64..127, up ..]."""
    if not gated:
        return torch.arange(m, device=dev)
    half = m // 2
    fb = torch.arange(m // 256, device=dev)[:, None, None]
    q = torch.arange(4, device=dev)[None, :, None]
    r = torch.arange(64, device=dev)[None, None, :]
    feat = fb * 128 + (q >> 1) * 64 + r
    return (feat + (q & 1) * half).reshape(-1)


def repack(w, sf, gated):
    """w [E, M, K/2] uint8 packed e2m1, sf [E, M, K/16] e4m3 -> (blob [E, NFB, KT, 4, 128, 16] uint8,
    sfblob [E, NFB, KT, 4, 32, 4, 2] uint8). A lossless permutation of the same bytes."""
    E, M, KB = w.shape
    dev = w.device
    sf = sf.view(torch.uint8)
    order = cta_row_order(M, gated, dev)
    row, col, srow = _blob_index(dev)
    nfb, kt = M // 256, KB // 32
    blobs, sfbs = [], []
    for e in range(E):
        we = w[e][order]                                         # [M, KB]
        se = sf[e][order]                                        # [M, K/16]
        g = torch.arange(M // 64, device=dev)
        k = torch.arange(kt, device=dev)
        rows = 64 * g[:, None, None, None] + row[None, None]     # (G, 1, 128, 16)
        cols = 32 * k[None, :, None, None] + col[None, None]     # (1, KT, 128, 16)
        bl = we[rows, cols]                                      # (G, KT, 128, 16)
        bl = bl.view(nfb, 4, kt, 128, 16).permute(0, 2, 1, 3, 4).contiguous()
        bidx = torch.arange(4, device=dev)
        i = torch.arange(2, device=dev)
        srows = 64 * g[:, None, None, None, None] + srow[None, None, :, None, None] + 8 * i[None, None, None, None, :]
        scols = 4 * k[None, :, None, None, None] + bidx[None, None, None, :, None]
        sb = se[srows, scols]                                    # (G, KT, 32, 4, 2)
        sb = sb.view(nfb, 4, kt, 32, 4, 2).permute(0, 2, 1, 3, 4, 5).contiguous()
        blobs.append(bl)
        sfbs.append(sb)
    return torch.stack(blobs), torch.stack(sfbs)


_mod = None


def load_module(src_path):
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a" if cap == (9, 0) else f"{cap[0]}.{cap[1]}"
        src = open(src_path).read()
        _mod = load_inline(
            name="muse_prefill_grouped_w4a16",
            cpp_sources="void moe_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, "
                        "torch::Tensor y, torch::Tensor tile_start, int64_t nfb, int64_t kt, int64_t kt_out, "
                        "int64_t mode, int64_t grid);",
            cuda_sources=src, functions=["moe_gemm"],
            extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo"], verbose=False)
    return _mod


@triton.jit
def tile_offsets_kernel(counts_ptr, tstart_ptr, BNC: tl.constexpr):
    e = tl.arange(0, 32)
    c = tl.load(counts_ptr + e)
    nt = (c + BNC - 1) // BNC
    incl = tl.cumsum(nt, 0)
    tl.store(tstart_ptr + e, incl - nt)
    tl.store(tstart_ptr + 1 + e, incl, mask=e == 31)


@triton.jit
def gather_tiled_kernel(x_ptr, slot_e_ptr, slot_pos_ptr, tstart_ptr, out_ptr, row_ptr,
                        D: tl.constexpr, BNC: tl.constexpr):
    """Each local route's token row x[t] -> its expert's token tile in the swizzled tiled layout;
    row_ptr[t, k] = global sorted row (tile * BN + r) or -1."""
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    x = tl.load(x_ptr + t * D + c)
    kt = c // 64
    cc = c % 64
    for k in tl.static_range(8):
        e = tl.load(slot_e_ptr + t * 8 + k)
        if e >= 0:
            pos = tl.load(slot_pos_ptr + t * 8 + k)
            tile = tl.load(tstart_ptr + e) + pos // BNC
            r = pos % BNC
            off = ((tile.to(tl.int64) * (D // 64) + kt) * BNC + r) * 64 + (((cc // 8) ^ (r % 8)) * 8) + cc % 8
            tl.store(out_ptr + off, x)
            tl.store(row_ptr + t * 8 + k, tile * BNC + r)
        else:
            tl.store(row_ptr + t * 8 + k, -1)


@triton.jit
def untile_rows_kernel(y_ptr, rows_ptr, out_ptr, D: tl.constexpr, BNC: tl.constexpr):
    """Debug helper: sorted row index -> row-major copy."""
    i = tl.program_id(0).to(tl.int64)
    g = tl.load(rows_ptr + i).to(tl.int64)
    tile = g // BNC
    r = g % BNC
    c = tl.arange(0, D)
    kt = c // 64
    cc = c % 64
    off = ((tile * (D // 64) + kt) * BNC + r) * 64 + (((cc // 8) ^ (r % 8)) * 8) + cc % 8
    tl.store(out_ptr + i * D + c, tl.load(y_ptr + off))


@triton.jit
def combine_tiled_kernel(y_ptr, rows_ptr, topk_ptr, pw_ptr, out_ptr, D: tl.constexpr, BNC: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, D)
    pw = tl.load(pw_ptr + c).to(tl.float32)
    acc = tl.zeros([D], tl.float32)
    for k in tl.static_range(8):
        g = tl.load(rows_ptr + t * 8 + k).to(tl.int64)
        if g >= 0:
            tile, r = g // BNC, g % BNC
            cc = c % 64
            off = ((tile * (D // 64) + c // 64) * BNC + r) * 64 + (((cc // 8) ^ (r % 8)) * 8) + cc % 8
            y = tl.load(y_ptr + off).to(tl.float32) * pw
            y *= tl.rsqrt(tl.sum(y * y, 0) / D + 1e-8)
            acc += tl.load(topk_ptr + t * 8 + k) * y
    tl.store(out_ptr + t * D + c, acc.to(tl.bfloat16))


def experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st):
    T = pooled.shape[0]
    dev = pooled.device
    max_tiles = triton.cdiv(T * 8 + 32 * (BN - 1), BN)
    tstart = torch.empty(33, dtype=torch.int32, device=dev)
    tile_offsets_kernel[(1,)](counts, tstart, BNC=BN)
    xn = torch.empty_like(pooled)
    _pre_expert_norm_kernel[(T,)](pooled, st['pre_expert'], xn, D=D, num_warps=4)
    xt = torch.empty((max_tiles * BN * D,), dtype=torch.bfloat16, device=dev)
    rows = torch.empty((T, 8), dtype=torch.int32, device=dev)
    gather_tiled_kernel[(T,)](xn, slot_e, slot_pos, tstart, xt, rows, D=D, BNC=BN, num_warps=4)
    inter = torch.empty_like(xt)
    y = torch.empty_like(xt)
    st['tiled_mod'].moe_gemm(st['blob13'], st['sfb13'], st['w13_gs'], xt, inter, tstart,
                             32, 64, 64, 1, st['sms'])
    st['tiled_mod'].moe_gemm(st['blob2'], st['sfb2'], st['w2_gs'], inter, y, tstart,
                             16, 64, 64, 0, st['sms'])
    out = torch.empty_like(pooled)
    combine_tiled_kernel[(T,)](y, rows, topk_w, st['post_expert'], out, D=D, BNC=BN, num_warps=4)
    return out
