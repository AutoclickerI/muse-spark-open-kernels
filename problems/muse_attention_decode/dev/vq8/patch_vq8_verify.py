"""patch(src) -> src: vq8 for the decode layer's verify rows where H200 labs measured a win: verify 1 sliding,
verify 16 full, verify 32 full (q_len 8; each with its own (stages, min_tiles)).

    python3 problems/muse_attention_decode/dev/vq8/patch_vq8_verify.py <cand.py> <out.py> [--rows 1s,16f,32f]

Adds vq8 (dev/vq8/vq8.cu, pinned by SHA-256 below) to a best7 decode-layer file (S10 ABI):
- `_vq8_module()`: load_inline, started in prepare's native pool.
- `_vq8_attention(qkv, data, setting)`: the layer ABI. qkv = (FP32 split partials [S, T, N], any S and split stride,
  unit stride along N) or (BF16 [1, T, N], 1). Returns BF16 [T, Q_SIZE] (RMSNorm x sigmoid(gate)) and writes
  the step's new K/V into the cache. Settings per row (Hopper; elsewhere 2 stages for 99 KB shared memory):
  verify 1 sliding (4, 1) = 33 CTAs of one 64-key tile; verify 16 / 32 full (6, 1).
- PDL: the main kernel is launched programmatic. Its producer warps read seq_lens, the page table and the
  cached K/V rows before griddepcontrol.wait (none are written by the preceding qkv GEMV); its consumers read
  qkv and write the new K/V slots only after the wait. Its combine kernel (also PDL) triggers
  launch_dependents at entry, so the O projection must griddepcontrol.wait before reading the output.
- `_vq8_prepare(device)`: after the attention tuner, vq8 runs the layer's own synthetic inputs for each row
  (FP32 partials, ctx 257 and 2305). It has to pass `_attention_store_agrees` (new K/V) and `_attention_agrees`
  against `_attention_reference`. Only then is the row routed (state['vq8_rows']). A failure prints
  VQ8_REJECT and the row keeps its S10 path.
- `_best7_attention` sends the routed rows to vq8; every other row is unchanged.

Evidence (proj-host lab prof/lab/vq8_layer.py: 4 FP32 partials, after a 37.7 MB projection stand-in with PDL,
cold L2, TASK gate incl. new K/V; vs the layer's qk_prep + public FA3 + norm_gate), this kernel, 44aa67ba029844d3:
verify 1 sliding st4mt1 15.30 us vs 16.83; verify 16 full st6mt1 153.9 vs 192.2; verify 32 full st6mt1 278.5 vs
381.4. Not routed (no win): verify 1 full 29.5 vs 28.5, verify 16 sliding 18.6 vs 16.0, verify 32 sliding 21.8 vs
21.8, and every q_len 1 row (attn_main). Earlier kernel f0a79d79bb614ccc: verify 1 sliding 14.40 vs 17.47.
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VQ8_CU = (HERE / 'vq8.cu').read_text()
VQ8_SHA = '69132eea2b81428eb23da3d00d7e7ef2bd8bdd90658004528e45792745250900'
ROWS = {'1s': ((1, 8, True), (4, 1)), '16f': ((16, 8, False), (6, 1)), '32f': ((32, 8, False), (6, 1))}

BLOCK = '''

# ---- vq8 (problems/muse_attention_decode/dev/vq8, vq8.cu SHA-256 {sha}): rows -> (stages, min_tiles) {rows} ----
VQ8_CUDA = {src}
VQ8_CPP = r"""
#include <cstdint>
#include <vector>
void vq8_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t);
std::vector<int64_t> vq8_resources(int64_t);
"""
_VQ8_MOD = None
_VQ8_ROWS = {rows}


def _vq8_module():
    global _VQ8_MOD
    if _VQ8_MOD is None:
        major, minor = torch.cuda.get_device_capability()
        arch = "90a" if major == 9 else f"{{major}}{{minor}}"
        _VQ8_MOD = load_inline(name=f"muse_vq8_{tag}_{{arch}}", cpp_sources=VQ8_CPP, cuda_sources=VQ8_CUDA,
                               functions=["vq8_forward", "vq8_resources"],
                               extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{{arch}},code=sm_{{arch}}"], verbose=False)
    return _VQ8_MOD


def _vq8_attention(qkv, data, setting):
    """vq8 on the layer ABI (see dev/vq8/patch_vq8_verify.py): BF16 [T, Q_SIZE], new K/V written to the cache."""
    partial, splits = qkv
    cfg = data['config']
    batch, length, window = cfg['batch'], cfg['q_len'], cfg['window']
    device = partial.device
    stages, min_tiles = setting if torch.cuda.get_device_capability(device)[0] == 9 else (2, setting[1])
    if partial.dtype == torch.bfloat16:
        assert splits == 1
        raw, nsplit, sstride = (partial[0] if partial.dim() == 3 else partial), 0, 0
    else:
        assert partial.dtype == torch.float32 and partial.stride(-1) == 1
        raw, nsplit, sstride = partial, splits, partial.stride(0)
    grid = _num_sms(device)
    segments = batch * (HKV // 2)
    part = torch.empty(((grid + segments) * 128 * 68 + 256,), device=device, dtype=torch.float32)
    out = torch.empty((batch * length, Q_SIZE), device=device, dtype=torch.bfloat16)
    invf = _inv_freq(cfg['rope_theta'], device) if window else None
    _vq8_module().vq8_forward(
        raw.data_ptr(), data['cache_loc'].data_ptr(), data['page_table'].data_ptr(), data['seq_lens'].data_ptr(),
        data['k_cache'].data_ptr(), data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        invf.data_ptr() if invf is not None else 0, part.data_ptr(), out.data_ptr(), batch, length, HQ, HKV,
        raw.stride(-2), data['page_table'].shape[1], cfg['page_size'], window, nsplit, min_tiles, grid, stages,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634, sstride, 1,
        torch.cuda.current_stream(device).cuda_stream)
    return out


def _vq8_prepare(device):
    """Rows routed to vq8 once it passes the layer's own attention gate (new K/V and output) on them."""
    errors = []
    try:
        for (batch, length, sliding), setting in _VQ8_ROWS.items():
            for ctx in (257, 2305):
                qkv, data = _attn_port_synthetic(batch, length, 2048 if sliding else 0, device, ctx)
                got = _vq8_attention(qkv, data, setting)
                _attention_store_agrees(qkv, data)
                ok, error, bound = _attention_agrees(got, _attention_reference(qkv, data),
                                                     _attention_reference(qkv, data, True))
                if not ok:
                    raise RuntimeError(f'vq8 {{(batch, length, sliding)}} ctx {{ctx}}: {{error:.5f}} > {{bound:.5f}}')
                errors.append(round(error, 5))
    except Exception as exc:
        print('VQ8_REJECT', str(exc).splitlines()[0][:200], flush=True)
        return {{}}
    print('VQ8_ROWS', _VQ8_ROWS, 'errors', errors, flush=True)
    return _VQ8_ROWS
'''

POOL_OLD = "        native_pool.submit(_layer_attn_ext)  # used by the attention tuner and the 'layer' rows\n"
POOL_NEW = POOL_OLD + "        native_pool.submit(_vq8_module)  # vq8 rows (q_len 8)\n"
PREP_OLD = "    state['attention_plan']=_tune_attention_portfolio(state,flush,device)\n"
PREP_NEW = PREP_OLD + "    state['vq8_rows']=_vq8_prepare(device)\n"
ROUTE_OLD = "def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n"
ROUTE_NEW = ROUTE_OLD + ("    vq8=state.get('vq8_rows',{}).get((cfg['batch'],cfg['q_len'],bool(cfg['window'])))\n"
                         "    if vq8 is not None:return _vq8_attention(qkv,data,vq8)\n")


def _sub(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f'expected one {old[:60]!r}, found {src.count(old)}')
    return src.replace(old, new)


def patch(src: str, rows=('1s', '16f', '32f')) -> str:
    sha = hashlib.sha256(VQ8_CU.encode()).hexdigest()
    if sha != VQ8_SHA:
        raise SystemExit(f'vq8.cu is {sha}, the checked cell pins {VQ8_SHA}')
    src = _sub(src, POOL_OLD, POOL_NEW)
    src = _sub(src, PREP_OLD, PREP_NEW)
    src = _sub(src, ROUTE_OLD, ROUTE_NEW)
    routed = dict(ROWS[r] for r in rows)
    return src + BLOCK.format(sha=sha, rows=repr(routed), src=repr(VQ8_CU), tag=sha[:12])


if __name__ == '__main__':
    rows = tuple(sys.argv[sys.argv.index('--rows') + 1].split(',')) if '--rows' in sys.argv else ('1s', '16f', '32f')
    out = patch(Path(sys.argv[1]).read_text(), rows)
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
