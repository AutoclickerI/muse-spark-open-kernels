M1_STREAM_MOD = None
M1_STREAM_HOPPER = torch.cuda.get_device_capability() == (9, 0)


def m1_stream_module():
    global M1_STREAM_MOD
    if M1_STREAM_MOD is None:
        import hashlib
        tag = hashlib.sha256(M1_STREAM_CUDA_SRC.encode()).hexdigest()[:12]
        M1_STREAM_MOD = load_inline(name='muse_m1_stream_' + tag, cpp_sources=M1_STREAM_CPP_SRC,
            cuda_sources=M1_STREAM_CUDA_SRC, functions=['run'],
            no_implicit_headers=True, with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo',
                *(['-gencode=arch=compute_90a,code=sm_90a'] if M1_STREAM_HOPPER else [])], verbose=False)
    return M1_STREAM_MOD


def m1_stream_kernel(data, bank, splits=4, pdl=False, scratch=False, grid=264):
    x = data['hidden_states'].contiguous()
    if x.shape != (1, 4096):
        raise ValueError('M1 streaming expert requires shape (1,4096)')
    if splits not in (1, 2, 4, 8):
        raise ValueError('M1 split K must be 1/2/4/8')
    if grid not in (0, 132, 264, 528, 1056):
        raise ValueError('M1 grid must be 0/132/264/528/1056')
    ids = data['topk_ids'].int().contiguous()
    tw = data['topk_weights'].float().contiguous()
    dev = x.device
    inter = torch.empty((8, 4096), device=dev, dtype=torch.bfloat16)
    parts = torch.empty((8, splits, 4096), device=dev)
    y = torch.empty((8, 4096), device=dev)
    sums = torch.empty((8, 256), device=dev)
    invnorm = torch.empty((8,), device=dev)
    counts = torch.empty((8 * 257 + 1,), device=dev, dtype=torch.int32)
    out = torch.empty_like(x)
    mod = bank['stream_mod']
    tensors = (x, ids, tw, bank['w13p'], bank['w2p'], bank['gs13'], bank['gs2'], bank['pnw'],
               inter, parts, y, sums, invnorm, counts, out)
    mod.run(*(tensor.data_ptr() for tensor in tensors), data['config']['local_start'], splits, pdl, grid,
            torch.cuda.current_stream(dev).cuda_stream)
    return (inter, parts, y, sums, invnorm, counts, out) if scratch else out
