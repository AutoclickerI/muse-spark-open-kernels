# Embedded into the lab payload. Only setup is shortened; every timed S11
# function and every selected expert/attention/projection gate is unchanged.
def tl_prepare_selected(host, weights):
    from concurrent.futures import ThreadPoolExecutor
    started=time.monotonic()
    host._BEST5_DEADLINE=started+235.
    device=weights['qkv_gate'].device
    builds=[host._layer_attn_ext,host._vq8_module,host._v9_recoded_module,
            host.nf_module,host.smallm_module,host._glue_cl_ext]
    if host._attn_port_hopper(device):builds.append(host._attn_port_ext)
    if host._fa3_on_device(device):builds.append(host._lt_module)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fn) for fn in builds]
        state=host._prepare_weights(weights)
        for future in futures:future.result()
    host._seed_v9_gemv_choices(state)
    flush=torch.empty(256<<20,device=device,dtype=torch.uint8)
    # Keep the original pinned projection constructors, Lt heuristic tables,
    # lossless layouts and their FP32/replay guards. No projection is substituted.
    state['proj_plan']=host._tune_projections(state,flush,device)
    host._tune_measured_lt(state,flush,device)
    state.pop('_proj_data',None);state.pop('_proj_stream',None)
    # All twelve BEST7_SETTINGS attention/expert choices are explicit. The
    # portfolio tuner and its unselected adapters cannot influence these calls.
    assert all(row[1]!='auto' and row[2]!='auto' for key,row in host.BEST7_SETTINGS.items() if key!='other')
    state['attention_plan']={(b,l,w):('layer',None) for b in (1,16,32) for l in (1,8) for w in (True,False)}
    state['vq8_rows']=host._vq8_prepare(device)
    assert state['vq8_rows']==host._VQ8_ROWS, 'S11 selected vq8 rows must pass their original gates'
    state['pdl_plan']={(t,p):True for t in host._TUNE_TOKENS for p in ('b1','b2')}
    state['prefix_plan']={t:('regular',True) for t in host._TUNE_TOKENS}
    state['moe_bank_raw']=state['moe_bank']
    state['moe_recoded_mod']=host._v9_recoded_module()
    state['moe_bank']=host._prepare_recoded_bank(weights,state['moe_bank_raw'],state['moe_recoded_mod'])
    state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event())
    host._s4_prepare_banks(state,weights)
    host._gate_split_prepare(state,device)
    # These are S11's original, strict, selected three-stream expert and
    # attention gates, including graph replays. They still raise on failure.
    host._best7_prepare_settings(state,weights,device)
    if torch.cuda.get_device_capability()==(9,0):
        for key,pin in host.PROJ_PINS.items():
            assert state['proj_plan'][key]==pin, ('pinned projection rejected',key,state['proj_plan'][key],pin)
    torch.cuda.synchronize()
    print('TL_SELECTED_PREPARE seconds=%.3f selected_gates=%d pins=%s'%
          (time.monotonic()-started,len(state['best7_gates']),repr(state['proj_plan'])),flush=True)
    return state
