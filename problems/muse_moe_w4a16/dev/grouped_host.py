import os
import triton
import triton.language as tl

_GROUPED_MOD = None


def _grouped_module():
    global _GROUPED_MOD
    if _GROUPED_MOD is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        arch = torch.cuda.get_device_capability()
        previous = os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' if arch == (9, 0) else f'{arch[0]}.{arch[1]}'
        try:
            _GROUPED_MOD = load_inline(
                name='muse_canonical_rs_' + hashlib.sha1(GROUPED_SRC.encode()).hexdigest()[:12],
                cpp_sources='void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\n'
                            'void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);',
                cuda_sources=GROUPED_SRC, functions=['route', 'gemm'],
                extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '--ptxas-options=-v'], verbose=False)
        finally:
            if previous is None:
                os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
            else:
                os.environ['TORCH_CUDA_ARCH_LIST'] = previous
    return _GROUPED_MOD


@triton.jit
def _combine_canonical(Y, POS, WEIGHTS, NORM, OUT, HIDDEN: tl.constexpr):
    token = tl.program_id(0)
    col = tl.arange(0, HIDDEN)
    # Preserve the effective BF16 norm weight used by the serving baseline.
    norm = (1.0 + tl.load(NORM + col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    acc = tl.full((HIDDEN,), 0, tl.float32)
    for slot in range(8):
        pos = tl.load(POS + token * 8 + slot)
        y = tl.load(Y + pos.to(tl.int64) * HIDDEN + col, pos >= 0, 0).to(tl.float32) * norm
        inv = tl.rsqrt(tl.sum(y * y, 0) / HIDDEN + 1e-8)
        acc += tl.load(WEIGHTS + token * 8 + slot) * (y * inv)
    tl.store(OUT + token * HIDDEN + col, acc.to(tl.bfloat16))


def prepare(weights):
    state = _prepare_native(weights)
    state['grouped_mod'] = _grouped_module()
    return state


def _grouped_call(data, state, fixed_n=0, stages=2):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    weights = data['topk_weights'].to(torch.float32).contiguous()
    cfg = data['config']
    assert cfg['local_count'] == 32 and cfg['d_hidden'] == 4096 and cfg['d_expert'] == 4096
    m, hidden = x.shape
    count = m * 8
    dev = x.device
    counts = torch.empty((32,), dtype=torch.int32, device=dev)
    offsets = torch.empty((33,), dtype=torch.int32, device=dev)
    cursors = torch.empty((32,), dtype=torch.int32, device=dev)
    pairs = torch.empty((count,), dtype=torch.int32, device=dev)
    positions = torch.empty((count,), dtype=torch.int32, device=dev)
    tasks = torch.empty((5 * 33,), dtype=torch.int32, device=dev)
    inter = torch.empty((count, hidden), dtype=torch.bfloat16, device=dev)
    y = torch.empty_like(inter)
    out = torch.empty_like(x)
    mod = state['grouped_mod']
    mod.route(ids, counts, offsets, cursors, pairs, positions, tasks, cfg['local_start'], fixed_n)
    grid = state['grid'] * 4
    mod.gemm(state['w13p'], state['gs13'], x, inter, pairs, offsets, tasks, 1, fixed_n, grid, stages)
    mod.gemm(state['w2p'], state['gs2'], inter, y, pairs, offsets, tasks, 0, fixed_n, grid, stages)
    _combine_canonical[(m,)](y, positions, weights, state['pnw'], out, HIDDEN=hidden, num_warps=8)
    return out


def kernel_fn(data, state):
    if data['hidden_states'].shape[0] < 64:
        return _native_call(data, state)
    return _grouped_call(data, state)
