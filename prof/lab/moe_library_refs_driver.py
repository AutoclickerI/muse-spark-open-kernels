"""Additional W4A16 library/phase measurements, separate from admissible kernel ranking."""

def lab_cutlass_bank(w):
    import flashinfer
    from flashinfer.fused_moe import interleave_moe_weights_for_sm90_mixed_gemm,interleave_moe_scales_for_sm90_mixed_gemm
    dev=w['w13'].device;bank={}
    for name,n in (('w13',8192),('w2',4096)):
        q=w[name]
        if name=='w13':q=lab_torch.cat((q[:,4096:],q[:,:4096]),1).contiguous()
        bank[name]=interleave_moe_weights_for_sm90_mixed_gemm(q,'fp4')
        # Reference-only MXFP4 weights: E8M0 group32 is what this installed kernel supports.
        # These cannot losslessly encode arbitrary E4M3 group16 NVFP4 scales.
        sf=lab_torch.full((32,n,128),121,device=dev,dtype=lab_torch.uint8)
        bank[name+'_s']=interleave_moe_scales_for_sm90_mixed_gemm(sf,32).view(lab_torch.int32)
    return bank


def lab_cutlass(d,b):
    import flashinfer
    out=flashinfer.cutlass_fused_moe(input=d['hidden_states'],token_selected_experts=d['topk_ids'],
        token_final_scales=d['topk_weights'],fc1_expert_weights=b['w13'],fc2_expert_weights=b['w2'],
        output_dtype=lab_torch.bfloat16,quant_scales=[b['w13_s'],b['w2_s']],
        ep_size=8,ep_rank=0,tp_size=1,tp_rank=0,use_w4_group_scaling=True,
        tune_max_num_tokens=1024,enable_pdl=False)
    return out[0] if isinstance(out,(list,tuple)) else out


def lab(deadline,weights):
    dev=next(iter(weights.values())).device
    if lab_torch.cuda.get_device_capability(dev)[0]>=12:
        # Validate the exact keyword ABI locally; SM90 arithmetic remains judge-only.
        import flashinfer
        original=flashinfer.cutlass_fused_moe
        # Calling FlashInfer on sm120 would select a different kernel than Hopper.
        import inspect
        kw=dict(input=lab_torch.empty(8,4096,device=dev,dtype=lab_torch.bfloat16),token_selected_experts=lab_torch.zeros(8,8,device=dev,dtype=lab_torch.int32),token_final_scales=lab_torch.ones(8,8,device=dev),fc1_expert_weights=lab_torch.empty(1,device=dev,dtype=lab_torch.uint8),fc2_expert_weights=lab_torch.empty(1,device=dev,dtype=lab_torch.uint8),output_dtype=lab_torch.bfloat16,quant_scales=[],ep_size=8,ep_rank=0,tp_size=1,tp_rank=0,use_w4_group_scaling=True,tune_max_num_tokens=1024,enable_pdl=False)
        inspect.signature(original).bind(**kw)
        return lab_local_glue(dev)+['LOCAL_LIBRARY cutlass signature bound; SM90 MXFP4 arithmetic unverified']
    out=['META reference-only library and V9_phase clock64; no dispatch from MXFP4/no-postnorm timings']
    rw,source=lab_router(weights,dev)
    if rw['pooler'].numel()==0:raise ValueError('proj host required for pooler')
    w=lab_experts(dev);flush=lab_torch.zeros((128<<20)//4,device=dev)
    native=v9_prepare(w)
    dsmap={(m,r):lab_datasets(rw,m,r,dev) for m in (8,16,32,128,256,512,753) for r in ('R','B')}
    # Compile/inventory optional references independently. Exceptions remain explicit.
    for name,init,fn in [('bf16ref',lambda:lab_bf16_bank(w),lab_bf16),('fi_mxfp4_ref',lambda:lab_cutlass_bank(w),lab_cutlass)]:
        if lab_time.monotonic()>deadline-20:
            out.append('BUDGET '+name);continue
        try:bank=init()
        except Exception as exc:lab_error(out,'init:'+name,exc);continue
        for (m,r),ds in dsmap.items():
            if lab_time.monotonic()>deadline-8:
                out.append('BUDGET '+name);break
            try:
                old=LabGraph(lambda d:v9_kernel_fn(d,native),ds[0]);new=LabGraph(lambda d,f=fn,b=bank:f(d,b),ds[0])
                got=new.replay();lab_torch.cuda.synchronize()
                if not bool(lab_torch.isfinite(got).all()):raise ValueError('nonfinite')
                us,raw,anchor=lab_pair(old,new,ds,flush)
                out.append(f'T {name} {m} {r} {us:.1f}')
                del old,new
            except Exception as exc:lab_error(out,f'{name}/{m}/{r}',exc)
        del bank
        lab_torch.cuda.empty_cache()
    if lab_time.monotonic()<deadline-25:
        try:
            tracebank=dict(native,mod=ph__module('trace'))
            for m in (8,16,32,128,256):
                for r in ('R','B'):
                    d=dsmap[m,r][0];mod=tracebank['mod'];stages=2
                    while stages>1 and mod.smem_needed(1,stages,m)>mod.smem_optin:stages-=1
                    grid=tracebank['grid']*mod.resident_blocks(1,stages,m)
                    trace=lab_torch.zeros((grid,10),device=dev,dtype=lab_torch.int64)
                    ph__kernel_v9(d,tracebank,stages,trace=trace);lab_torch.cuda.synchronize()
                    rows=trace.cpu().tolist()
                    stats=[]
                    for i in range(7):
                        vals=[a[i] for a in rows if a[i]];stats.append(f'{lab_stats.median(vals):.0f}/{max(vals)}' if vals else '0/0')
                    out.append(f'CY {m} {r} '+','.join(stats))
        except Exception as exc:lab_error(out,'trace',exc)
    out.append('CY order=route,W13+SwiGLU,wait,barrier,W2,norm,combine median/max clock64 cycles; not additive wall times')
    out.append('LIB vllm absent; Machete absent in sgl_kernel; Marlin supports NVFP4; FlashInfer W4A16=E8M0/group32 MXFP4; FP4-activation kernels excluded')
    out.append(f'END left_s={deadline-lab_time.monotonic():.1f}')
    return out
