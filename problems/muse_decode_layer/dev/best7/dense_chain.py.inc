DENSE_CHAIN_CPP = r'''
#include <torch/extension.h>
void island(std::vector<torch::Tensor> a, int64_t m, int64_t n, int64_t splits, int64_t nt,
            int64_t grid, int64_t prefetch, bool prefix, bool pdl, bool route);
'''
_DENSE_CHAIN_MOD = None


def _dense_chain_module():
    global _DENSE_CHAIN_MOD
    if _DENSE_CHAIN_MOD is None:
        import hashlib
        tag = hashlib.sha256(DENSE_CHAIN_CUDA.encode()).hexdigest()[:12]
        _DENSE_CHAIN_MOD = load_inline(name='muse_dense_chain_' + tag,
            cpp_sources=DENSE_CHAIN_CPP, cuda_sources=DENSE_CHAIN_CUDA,
            functions=['island'], extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
    return _DENSE_CHAIN_MOD


def _dense_chain_prepare(state, weights, device):
    state['dense_chain_mod'] = _dense_chain_module()
    state['dense_chain_grid'] = torch.cuda.get_device_properties(device).multi_processor_count


def _dense_chain_call(attn, stream, state, data):
    t = attn.shape[0]; dev = attn.device
    partial = torch.empty((1, t, 8192), device=dev, dtype=torch.float32)
    pr_out = torch.empty((t, 4608), device=dev, dtype=torch.float32)
    stream_out = torch.empty_like(stream)
    hidden = torch.empty((t, 8192), device=dev, dtype=torch.bfloat16)
    zbf = torch.empty((t, 4096), device=dev, dtype=torch.bfloat16)
    zi = torch.empty((t, 8), device=dev, dtype=torch.int32)
    zf = torch.empty((t, 8), device=dev, dtype=torch.float32)
    cnt = torch.empty((2 + 5 * t,), device=dev, dtype=torch.int32)
    local = data['local_start']
    args = [attn, state['o'], state['pool_router'], state['pool_router'], stream,
            state['alpha1'], state['beta1'], state['pre_ffn_w'], state['post_ffn_w'],
            state['router_bias'], state['pre_expert_w'], local, partial, pr_out,
            stream_out, hidden, zbf, zi, zi, cnt, zf]
    cnt.zero_()
    prefetch = min(16 << 20, state['pool_router'].numel() * 2)
    state['dense_chain_mod'].island(args, t, 4608, 1, 4, state['dense_chain_grid'],
                                    prefetch, True, False, False)
    return hidden, stream_out, pr_out
