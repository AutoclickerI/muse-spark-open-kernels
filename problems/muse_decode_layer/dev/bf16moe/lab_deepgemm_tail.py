"""Tail bundled by build_lab.py, after prefixed V9 and router fixture helpers."""


def _bf_expand(packed, scale):
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=packed.device)
    result = torch.empty((*packed.shape[:-1], packed.shape[-1]*2), dtype=torch.bfloat16, device=packed.device)
    for e in range(packed.shape[0]):
        code = torch.stack((packed[e]&15, packed[e]>>4), -1).flatten(-2)
        values = grid[(code&7).long()] * torch.where(code&8 != 0, -1., 1.)
        values = values.view(*scale[e].shape, 16)*scale[e].float().unsqueeze(-1)
        result[e] = values.flatten(-2).to(torch.bfloat16)
    return result


def _bf_lab_time(fn, flush, repeats=16):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        flush.sum()
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b)*1000)
    graph.reset()
    del output
    return statistics.median(times)


import triton
import triton.language as tl

@triton.jit
def _bf_lab_swiglu(H,I,GS,C,START,M:tl.constexpr,PACKED:tl.constexpr):
    r=tl.program_id(0)
    e=tl.program_id(1)
    count=tl.load(C+e)
    if r<count:
        if PACKED:
            base=tl.load(START+e)+r
        else:
            base=e*M+r
        d=tl.arange(0,4096)
        sc=tl.load(GS+e)
        gate=tl.load(H+base*8192+d).to(tl.float32)*sc
        up=tl.load(H+base*8192+4096+d).to(tl.float32)*sc
        out=gate*tl.sigmoid(gate)*up
        tl.store(I+base*4096+d,out)

def lab(deadline, weights):
    import deep_gemm as dg
    lines=[f'LAB_DG torch={torch.__version__} sm={torch.cuda.get_device_capability()}']
    w=_bf__lab_make_weights(5353)
    native=_bf_v9_v9_prepare({**w,'post_norm_w':w['post_expert_norm']})
    w13,w2=_bf_expand(w['w13'],w['w13_sf']),_bf_expand(w['w2'],w['w2_sf'])
    flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8)
    for m in (128,256):
        data=_bf_make_fixture(w,m,5353+m)
        ids,x=data['topk_ids'],data['hidden_states']
        counts=torch.bincount(ids[ids>=0],minlength=32).cpu().tolist()
        lines.append(f'LAB_ROUTE M={m} counts={counts}')
        v9_us=_bf_lab_time(lambda:_bf_v9_v9_kernel_fn(data,native),flush)
        lines.append(f'LAB_V9 M={m} full_us={v9_us:.3f}')
        for threshold in (16,32,64):
            hot=[c if c>=threshold else 0 for c in counts]
            cm=torch.tensor(hot,device='cuda',dtype=torch.int32)
            a=torch.zeros((32,m,4096),device='cuda',dtype=torch.bfloat16)
            for e in range(32):
                if hot[e]:
                    pair=(ids==e).nonzero();a[e,:len(pair)]=x[pair[:,0]]
            cold=dict(data)
            cold['topk_ids']=torch.where((ids>=0)&(cm[ids.clamp_min(0).long()]>0),-1,ids)
            cold_native={**native,'grid':max(1,native['grid']//2)}
            cold_us=_bf_lab_time(lambda:_bf_v9_v9_kernel_fn(cold,cold_native),flush)
            for dtype in (torch.bfloat16,torch.float32):
                if time.monotonic()>deadline-15:
                    lines.append('LAB_BUDGET');return lines
                try:
                    h=torch.empty((32,m,8192),device='cuda',dtype=dtype)
                    inter=torch.empty((32,m,4096),device='cuda',dtype=torch.bfloat16)
                    y=torch.empty((32,m,4096),device='cuda',dtype=dtype)
                    expected=max(16,sum(hot)//max(1,sum(c>0 for c in hot)))
                    def masked():
                        dg.m_grouped_bf16_gemm_nt_masked(a,w13,h,cm,expected)
                        _bf_lab_swiglu[(m,32)](h,inter,w['w13_gs'],cm,cm,M=m,PACKED=False,num_warps=4)
                        dg.m_grouped_bf16_gemm_nt_masked(inter,w2,y,cm,expected)
                        return y
                    us=_bf_lab_time(masked,flush,8)
                    lines.append(f'LAB_DG_MASK M={m} threshold={threshold} expected={expected} dtype={dtype} dense_us={us:.3f} cold_us={cold_us:.3f}')
                except Exception as exc:
                    lines.append(f'LAB_DG_MASK_ERROR M={m} threshold={threshold} dtype={dtype} {str(exc)[:160]}')
            # Prefix-sum contiguous format packs only touched groups, padded128.
            padded=[((c+127)//128)*128 for c in hot]
            ends=torch.tensor(padded,device='cuda',dtype=torch.int32).cumsum(0).int()
            starts=torch.cat((torch.zeros(1,device='cuda',dtype=torch.int32),ends[:-1])).contiguous()
            cap=m*8+32*128
            ax=torch.zeros((cap,4096),device='cuda',dtype=torch.bfloat16)
            eid=torch.zeros(cap,device='cuda',dtype=torch.long)
            pos=0
            for e in range(32):
                if hot[e]:
                    pair=(ids==e).nonzero();ax[pos:pos+hot[e]]=x[pair[:,0]]
                    eid[pos:pos+padded[e]]=e;pos+=padded[e]
            for dtype in (torch.bfloat16,torch.float32):
                if time.monotonic()>deadline-15:
                    lines.append('LAB_BUDGET');return lines
                try:
                    hx=torch.empty((cap,8192),device='cuda',dtype=dtype)
                    ix=torch.empty((cap,4096),device='cuda',dtype=torch.bfloat16)
                    yx=torch.empty((cap,4096),device='cuda',dtype=dtype)
                    def psum():
                        dg.m_grouped_bf16_gemm_nt_contiguous(ax,w13,hx,ends,use_psum_layout=True)
                        _bf_lab_swiglu[(m,32)](hx,ix,w['w13_gs'],cm,starts,M=m,PACKED=True,num_warps=4)
                        dg.m_grouped_bf16_gemm_nt_contiguous(ix,w2,yx,ends,use_psum_layout=True)
                        return yx
                    us=_bf_lab_time(psum,flush,8)
                    lines.append(f'LAB_DG_PSUM M={m} threshold={threshold} dtype={dtype} padded={pos} dense_us={us:.3f} cold_us={cold_us:.3f}')
                except Exception as exc:
                    lines.append(f'LAB_DG_PSUM_ERROR M={m} threshold={threshold} dtype={dtype} {str(exc)[:160]}')
    return lines
