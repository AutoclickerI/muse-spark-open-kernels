from pathlib import Path
p=Path('problems/muse_prefill_layer')
s=(p/'cand_v6.py').read_text()
s=s.replace('GemmType.INDEXED', 'GemmType.GROUPED_MASKED')
s=s.replace('moe = _experts_humming(pooled, topk_w, slot_e, st)', '''if T < 2048:
        moe = _experts_humming_masked(pooled, topk_w, slot_e, slot_pos, counts, st)
    else:
        moe = _experts_dequant(pooled, topk_w, slot_e, slot_pos, counts, st)''')
s+='''

@triton.jit
def _fixed_offsets_kernel(Starts, T: tl.constexpr):
    e=tl.arange(0,32)
    tl.store(Starts+e,e*T)


def _experts_humming_masked(pooled, topk_w, slot_e, slot_pos, counts, st):
    T=pooled.shape[0]
    stride=triton.cdiv(T,128)*128
    starts=torch.empty(32,device=pooled.device,dtype=torch.int32)
    _fixed_offsets_kernel[(1,)](starts,T=stride)
    a=torch.empty((32*stride,4096),device=pooled.device,dtype=torch.bfloat16)
    rows=torch.empty((T,8),device=pooled.device,dtype=torch.int32)
    _gather_norm_kernel[(T,)](pooled,st['pre_expert'],slot_e,slot_pos,starts,a,rows,D=4096,num_warps=4)
    h=torch.empty((32*stride,8192),device=pooled.device,dtype=torch.bfloat16)
    inter=torch.empty_like(a)
    y=torch.empty_like(a)
    bank=st['humming']
    routes,gemms=bank.plan(T)
    humming_ops.launch_kernel(configs=gemms['w13'][1],inputs=a,outputs=h,**bank.weight('w13'),
            locks=bank.locks,expert_layout=counts,valid_shape_m=routes)
    _swiglu_kernel[(32,triton.cdiv(T,4))](h,inter,starts,counts,st['ones'],BR=4,D=4096,num_warps=8)
    humming_ops.launch_kernel(configs=gemms['w2'][1],inputs=inter,outputs=y,**bank.weight('w2'),
            locks=bank.locks,expert_layout=counts,valid_shape_m=routes)
    out=torch.empty_like(pooled)
    _combine_kernel[(T,)](y,rows,slot_e,topk_w,st['ones'],st['post_expert'],out,D=4096,num_warps=4)
    return out
'''
s=s.replace("st['humming'] = _Experts(weights, 0, 32)", "st['humming'] = _Experts(weights, 0, 32)\n    st['ones'] = torch.ones(32, device=weights['w13'].device, dtype=torch.float32)")
(p/'cand_v8.py').write_text(s)
