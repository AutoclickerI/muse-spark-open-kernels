def lab(deadline,weights,state=None):
    started=time.monotonic();deadline=min(deadline,started+230)
    st=gb_state(weights);dev=weights['o'].device
    st['inv_freq']=(1./(500000.**(torch.arange(0,64,2,device=dev).float()/64))).contiguous()
    _gbc__seed_v9_gemv_choices(st)
    flush=torch.zeros((1<<30)//4,device=dev,dtype=torch.float32)
    lines=['GB_META current=cand_r3_best_812f5f36 dense=proj_host norm_gates=synthetic_model_distribution cold1GiB graph paired20 fresh3; route=sigmoid_biased_top8; BF16_boundaries_preserved']
    # Original best's own projection, Lt, PDL and prefix selection, without MoE/attention setup.
    try:
        st['proj_plan']=_gbc__tune_projections(st,flush,dev)
        _gbc__tune_measured_lt(st,flush,dev)
        st['pdl_plan']=_gbc__tune_pdl(st,flush,dev)
        st['prefix_plan']=_gbc__tune_prefix(st,flush,dev)
    except Exception as exc:
        # Missing optional incumbent setup cannot produce a misleading comparison.
        return lines+[f'GB_REJECT control_setup {type(exc).__name__}:{str(exc)[-180:]}']
    callbacks=gb_callbacks(st,_gbc_project,_gbc_boundary,_gbc_route_and_norm,_gbc_o_and_pool)
    lines.append(f'GB_CONTROL_SETUP elapsed_s={time.monotonic()-started:.3f}')
    for m in GB_TOKENS:
        names=('o','pool_router','unpooler_slice' if m<=8 else 'unpooler')
        lines.append(f'GB_CONTROL {m} proj={[st["proj_plan"].get((m,n),"custom") for n in names]} prefix={st["prefix_plan"].get(m)}')
    lines+=gb_prepare(st,callbacks,deadline)
    lines.append(f'GB_LAB_END elapsed_s={time.monotonic()-started:.3f}')
    errors=[r for r in lines if r.startswith('GB_REJECT')]
    compact=[r for r in lines if not r.startswith('GB_REJECT')]
    compact += [r[:180] for r in errors[:6]]
    if errors:compact.append(f'GB_REJECT_COUNT {len(errors)} sampled={min(6,len(errors))}')
    return compact
