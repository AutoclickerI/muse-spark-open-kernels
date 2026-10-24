"""Probe vLLM/SGLang Humming W4A16 on the judge's real NVFP4 bank.
Measures the library's indexed W4A16 kernels at serving-shaped token counts.
No result is used by the scored host kernel; this is lab-only evidence.
"""
def lab(deadline, weights, state=None):
    import time
    import torch
    out = []
    try:
        saved = __import__('os').environ.get('HUMMING_DISABLE_PARALLEL_BUILD')
        __import__('os').environ['HUMMING_DISABLE_PARALLEL_BUILD'] = '1'
        from humming import ops as h_ops
        from humming.config import GemmType, WeightScale2Type
        from humming.kernel.humming import HummingKernel
        from humming.tune import get_heuristics_config
        from humming.layer import HummingMethod
        from humming.schema.modelopt import ModeloptNvfp4WeightSchema
        from sglang.kernels.ops.activation.activation import silu_and_mul
        from sglang.kernels.ops.moe.fused_moe_triton import moe_align_block_size
        if saved is None:
            __import__('os').environ.pop('HUMMING_DISABLE_PARALLEL_BUILD', None)
        else:
            __import__('os').environ['HUMMING_DISABLE_PARALLEL_BUILD'] = saved

        dev = weights['w13'].device
        d_i = weights['w13'].shape[1] // 2
        d_h = weights['w2'].shape[1]
        sl = slice(0, 32)
        layer = torch.nn.Module()
        metas = {}
        for name, n, k, stacks in (('w13', 2*d_i, d_h, [d_i, d_i]), ('w2', d_h, d_i, [d_h])):
            ck = {'weight': weights[name][sl].contiguous(),
                  'weight_scale': weights[name + '_sf'][sl].contiguous(),
                  'weight_scale_2': weights[name + '_gs'][sl].contiguous()}
            schema, tensors = ModeloptNvfp4WeightSchema().convert_humming(
                tensors=ck, shape_n_stacks=stacks, shape_k_stacks=[k],
                param_dtype=torch.bfloat16, num_experts=32)
            if schema.weight_scale_2_type != WeightScale2Type.TENSOR:
                raise RuntimeError(f'{name}: non-tensor global scale {schema.weight_scale_2_type}')
            for key, value in tensors.items():
                setattr(layer, f'{name}_{key}', torch.nn.Parameter(value, requires_grad=False))
            HummingMethod.prepare_layer_meta(layer=layer, shape_n=n, shape_k=k,
                weight_schema=schema, num_experts=32, pad_n_to_multiple=256,
                pad_k_to_multiple=128, torch_dtype=torch.bfloat16, sublayer_name=name)
            HummingMethod.transform_humming_layer(layer, sublayer_name=name)
            metas[name] = getattr(layer, 'humming_metas')[name]
        locks = torch.zeros(1024, dtype=torch.int32, device=dev)
        weights_for = lambda name: {
            'weight': getattr(layer, f'{name}_weight'),
            'weight_scale': getattr(layer, f'{name}_weight_scale'),
            'weight_scale_2': getattr(layer, f'{name}_weight_scale_2')}
        configs = {}
        for name in ('w13', 'w2'):
            needed = {}
            for bs in (1, 8, 16, 32, 128, 256):
                routes = max(1, bs * 8 * 32 // 256)
                hcfg = get_heuristics_config(layer_config=metas[name], shape_m=routes,
                    use_f16_accum=False, use_batch_invariant=False, gemm_type=GemmType.INDEXED)
                cfg = HummingKernel.prepare_kernels(
                    metas[name].to_str(),
                    __import__('json').dumps({'use_f16_accum': False, 'gemm_type': GemmType.INDEXED.value}),
                    [[0, 1, hcfg]])
                if cfg.numel() == 0:
                    raise RuntimeError(f'{name}: empty Humming launch config')
                configs[name] = cfg.view(-1, 4)[0].to(device='cpu')
        for bs in (1, 8, 16, 32, 128, 256):
            x = torch.randn((bs, d_h), device=dev, dtype=torch.bfloat16)
            ids = (torch.arange(bs * 8, device=dev, dtype=torch.int32).view(bs, 8) % 32).contiguous()
            sorted_ids, experts, n_pad = moe_align_block_size(ids, 16, 32)
            h = torch.empty((bs * 8, 2*d_i), device=dev, dtype=torch.bfloat16)
            a = torch.empty((bs * 8, d_i), device=dev, dtype=torch.bfloat16)
            y = torch.empty((bs * 8, d_h), device=dev, dtype=torch.bfloat16)
            for _ in range(3):
                h_ops.launch_kernel(configs=configs['w13'], inputs=x, outputs=h, **weights_for('w13'),
                    locks=locks, sorted_ids=sorted_ids, expert_ids=experts, num_tokens_padded=n_pad,
                    top_k=8, valid_shape_m=bs*8*32//256)
                silu_and_mul(h, a, expert_ids=ids.view(-1), expert_step=1)
                h_ops.launch_kernel(configs=configs['w2'], inputs=a, outputs=y, **weights_for('w2'),
                    locks=locks, sorted_ids=sorted_ids, expert_ids=experts, num_tokens_padded=n_pad,
                    top_k=1, valid_shape_m=bs*8*32//256)
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(10):
                h_ops.launch_kernel(configs=configs['w13'], inputs=x, outputs=h, **weights_for('w13'),
                    locks=locks, sorted_ids=sorted_ids, expert_ids=experts, num_tokens_padded=n_pad,
                    top_k=8, valid_shape_m=bs*8*32//256)
                silu_and_mul(h, a, expert_ids=ids.view(-1), expert_step=1)
                h_ops.launch_kernel(configs=configs['w2'], inputs=a, outputs=y, **weights_for('w2'),
                    locks=locks, sorted_ids=sorted_ids, expert_ids=experts, num_tokens_padded=n_pad,
                    top_k=1, valid_shape_m=bs*8*32//256)
            ev1.record(); ev1.synchronize()
            out.append(f'HUMMING_PROBE bs={bs} routes={bs*8*32//256} us_pair={(ev0.elapsed_time(ev1)*100.0):.3f}')
            if time.monotonic() > deadline:
                break
    except BaseException as e:
        out.append(f'HUMMING_PROBE_ERROR {type(e).__name__}: {e}')
    return out
