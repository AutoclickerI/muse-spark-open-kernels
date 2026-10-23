PERSIST_CPP = '''
#include <torch/extension.h>
void island(std::vector<torch::Tensor> a, int64_t m, int64_t n, int64_t splits, int64_t nt,
            int64_t grid, int64_t prefetch, bool prefix, bool pdl);
'''
_PERSIST_MOD = None
PERSIST_PREFETCH = True
PERSIST_MAX_T = 32


def _persistent_module():
    global _PERSIST_MOD
    if _PERSIST_MOD is None:
        import hashlib
        tag = hashlib.sha256(PERSIST_CUDA.encode()).hexdigest()[:12]
        _PERSIST_MOD = load_inline(name='muse_dense_islands_' + tag,
            cpp_sources=PERSIST_CPP, cuda_sources=PERSIST_CUDA, functions=['island'],
            extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo'], verbose=False)
    return _PERSIST_MOD

_prepare_before_persistent = prepare
_kernel_before_persistent = kernel_fn


def prepare(weights):
    state = _prepare_before_persistent(weights)
    state['persistent_mod'] = _persistent_module()
    state['dense_grid'] = torch.cuda.get_device_properties(weights['o'].device).multi_processor_count
    return state


def _island(inp, stream, state, data, counters, prefix, stamps=None):
    t = inp.shape[0]
    dev = inp.device
    n = HIDDEN if prefix or t > 8 else UNPOOL_SLICE
    splits = 1 if prefix or n == HIDDEN else 4
    nt = 1 if t <= 8 else 2 if t <= 16 else 4
    partial = torch.empty((splits,t,n), dtype=torch.float32, device=dev)
    # Prefix's pooler and router partials remain FP32 until their independent consumers.
    pr = torch.empty((t,4608) if prefix else (0,), dtype=torch.float32, device=dev)
    stream_out = torch.empty_like(stream)
    hidden = torch.empty((t,HIDDEN), dtype=torch.bfloat16, device=dev)
    x = torch.empty((t,D_EXPERT) if prefix else (0,), dtype=torch.bfloat16, device=dev)
    ids = torch.empty((t,TOP_K) if prefix else (0,), dtype=torch.int32, device=dev)
    lids = torch.empty_like(ids)
    wts = torch.empty(ids.shape, dtype=torch.float32, device=dev)
    pref_bytes = min(n * (1024 if prefix else 4096) * 2, 16 << 20) if PERSIST_PREFETCH else 0
    args = [inp,state['o'],state['pool_router'],state['unpooler'],
        stream,state['alpha1' if prefix else 'alpha2'],state['beta1' if prefix else 'beta2'],
        state['pre_ffn_w' if prefix else 'next_pre_attn_w'],state['post_ffn_w'],
        state['router_bias'],state['pre_expert_w'],data['local_start'],partial,pr,stream_out,
        hidden,x,ids,lids,counters,wts]
    if stamps is not None: args.append(stamps)
    state['persistent_mod'].island(args,t,n,splits,nt,state['dense_grid'],pref_bytes,prefix,PDL)
    return (x,ids,wts,lids,stream_out) if prefix else (hidden,stream_out)


def kernel_fn(data, state):
    if data['config']['tokens'] > PERSIST_MAX_T or state['dense_grid'] < 128:
        return _kernel_before_persistent(data,state)
    # One fixed-size zero for all grid barriers; no data-dependent allocation.
    counters = torch.zeros((2 * (2 + 5 * data['config']['tokens']),), dtype=torch.int32, device=data['stream'].device)
    qkv = gemv(data['attn_input'],state['qkv_gate'])
    attn = stage_attention(qkv,data,state)
    x,ids,wts,lids,stream1 = _island(attn,data['stream'],state,data,counters[:2+5*data['config']['tokens']],True)
    moe = experts_v4(x,wts,lids,state)
    hidden,stream2 = _island(moe,stream1,state,data,counters[2+5*data['config']['tokens']:],False)
    return hidden,stream2,ids
