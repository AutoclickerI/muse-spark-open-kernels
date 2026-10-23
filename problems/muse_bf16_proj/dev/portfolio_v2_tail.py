# Appended by build_portfolio_v2.py; not a standalone submission.
def _v2_run(data,state,config,scratch):
    if config is None:
        return _small(data,state)
    x=data['x']; layer=data['config']['layer']; n=data['config']['n']
    key=(layer,x.shape[0],n,x.shape[1])
    if config[0] not in ('pair','lt_nn'):
        return _launch_hop(x,state,layer,n,config,scratch)
    out=torch.empty((x.shape[0],n),device=x.device,dtype=torch.float32 if layer=='router' else torch.bfloat16)
    if config[0]=='pair':
        w=state['router_pair'] if layer=='router' else state[layer]
        state['mod'].hopper_pair(w,x,out,*scratch,*config[1:])
    else:
        state['mod'].lt_nn_run(state['lt_transposed'][layer],x,out,state['lt_workspace'],state['lt_nn_algorithms'][key],config[1])
    return out


def _v2_shared_bytes(layer,config):
    bt,bk,st=config[:3]
    return max(st*(64*(2 if layer=='router' else 1)+bt)*bk*2+st*16,bt*68*4)


def prepare(weights: Weights):
    import time
    started=time.monotonic();deadline=started+240
    state=_prepare_small(weights)
    state.update(choices={},scratch={},lt_algorithms={},lt_nn_algorithms={})
    state['router_pair']=torch.cat([state['router_hi'],state['router_lo']],dim=1).contiguous()
    state['lt_transposed']={layer:w.T.contiguous() for layer,w in weights.items() if layer!='router'}
    device=state['router_hi'].device
    state['lt_workspace']=torch.empty(32*1024*1024,device=device,dtype=torch.uint8)
    flush=torch.zeros(128*1024*1024,device=device,dtype=torch.uint8)
    props=torch.cuda.get_device_properties(device)
    for layer in ('router','qkv_gate','pooler','unpooler','o'):
        n,k=weights[layer].shape
        for m in (128,256):
            if time.monotonic()>=deadline:
                print(f'PORTFOLIO_V2_BUDGET {layer} m={m}',flush=True);continue
            x=torch.randn((m,k),device=device,dtype=torch.bfloat16)
            data={'x':x,'config':{'layer':layer,'n':n,'k':k,'m':m}}
            key=(layer,m,n,k)
            expected=x.double()@weights[layer].double().T
            scratch=(torch.empty(m*n*17,device=device,dtype=torch.float32),
                     torch.zeros((n//64)*((m+63)//64),device=device,dtype=torch.uint32))
            # The old heuristic indices/SM90 kernels are device-specific. On sm_120,
            # their Python ABI is forced separately with portable stand-ins.
            old=H200_WINNERS[(layer,m)] if _hopper() else None
            if old and old[0]=='lt':
                out=torch.empty((m,n),device=device,dtype=torch.bfloat16)
                try:
                    codes=state['mod'].lt_options(state[layer],x,out)
                    if old[1]>=codes.shape[0]:raise ValueError(f'H200 Lt winner unavailable: {old}')
                    state['lt_algorithms'][key]=codes
                except Exception as exc:
                    print(f'V2_REJECT {layer} m={m} legacy Lt: {str(exc)[:180]}',flush=True);old=None
            base_graph,base_out=_graph(lambda:_v2_run(data,state,old,scratch))
            configs=[]
            for config in FUSED_CONFIGS.get((layer,m),[]):
                if _v2_shared_bytes(layer,config)<=props.shared_memory_per_block_optin:
                    configs.append(('pair',)+tuple(config))
            if layer!='router':
                try:
                    nn_codes=state['mod'].lt_nn_options(state['lt_transposed'][layer],x,base_out)
                    state['lt_nn_algorithms'][key]=nn_codes
                    configs += [('lt_nn',i) for i in range(nn_codes.shape[0])]
                except Exception as exc:print(f'V2_REJECT {layer} m={m} NN setup: {str(exc)[:180]}',flush=True)
            trials=[]
            for config in configs:
                if time.monotonic()>=deadline:break
                try:
                    graph,out=_graph(lambda c=config:_v2_run(data,state,c,scratch))
                    error=(out.double()-expected).norm()/expected.norm()
                    limit=1e-4 if layer=='router' else 1e-2
                    if not bool(torch.isfinite(out).all()) or float(error)>limit:
                        raise ValueError(f'incorrect {config}: relative error {float(error)}')
                    trials.append((_cold_times([graph],flush,3)[0],config))
                except Exception as exc:print(f'V2_REJECT {layer} m={m} {config}: {str(exc)[:180]}',flush=True)
            trials.sort(key=lambda row:row[0]);selected=old
            # Pair finalists independently so an invalid finalist cannot abort
            # validation of the remaining methods or remove the proven control.
            best_ratio=1.0
            for _,config in trials[:3]:
                if time.monotonic()>=deadline:break
                try:
                    graph,out=_graph(lambda c=config:_v2_run(data,state,c,scratch))
                    error=(out.double()-expected).norm()/expected.norm()
                    if not bool(torch.isfinite(out).all()) or float(error)>limit:raise ValueError('final replay disagreement')
                    base_us,us=_cold_times([base_graph,graph],flush,9)
                    if us/base_us<best_ratio:best_ratio=us/base_us;selected=config
                except Exception as exc:print(f'V2_REJECT_FINAL {layer} m={m} {config}: {str(exc)[:180]}',flush=True)
            state['choices'][key]=selected
            state['scratch'][key]=scratch if selected and selected[0] not in ('lt','lt_nn') else None
            print(f'PORTFOLIO_V2 {layer} m={m} chosen={selected} ratio={best_ratio:.4f} top={trials[:3]}',flush=True)
    print(f'PORTFOLIO_V2 prepare_s={time.monotonic()-started:.2f} h200_job={H200_EVIDENCE["job"]}',flush=True)
    return state


def kernel_fn(data: Inputs,state)->torch.Tensor:
    x=data['x'];config=data['config']
    key=(config['layer'],x.shape[0],config['n'],x.shape[1])
    selected=state['choices'].get(key)
    return _v2_run(data,state,selected,state['scratch'].get(key))

