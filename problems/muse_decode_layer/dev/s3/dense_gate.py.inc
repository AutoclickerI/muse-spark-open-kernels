# Admit the persistent chain against the actual row's selected projection/glue.
_s3_chain_prepare_base = _dense_chain_prepare

def _dense_chain_prepare(state,weights,device):
    _s3_chain_prepare_base(state,weights,device)
    for key,plan in state['best7_plans'].items():
        if key=='other' or not plan['dense_chain']:continue
        batch,length,window=key;tokens=batch*length
        gen=torch.Generator(device=device).manual_seed(78219+tokens+window)
        attn=torch.randn((tokens,1024),device=device,dtype=torch.bfloat16,generator=gen)
        stream=torch.randn((tokens,8192),device=device,dtype=torch.float32,generator=gen)
        cfg=dict(tokens=tokens);data=dict(config=cfg,local_start=torch.zeros(1,device=device,dtype=torch.int32))
        sync=None
        if plan['glue']=='best5':
            raw=(torch.empty if _best7_old_chunks(tokens)==1 else torch.zeros)((15*tokens,),device=device,dtype=torch.int32)
            sync=(raw[3*tokens:].view(torch.float32).view(3,tokens,4)[0],raw[:3*tokens].view(3,tokens)[0])
        def incumbent():
            a=_best7_project(attn,'o',state,plan)
            h,s=_best7_boundary(a,stream,state,plan,'b1',sync)
            p,n=_best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
            return h,s,p.float().sum(0)
        def candidate():return _dense_chain_call(attn,stream,state,data)
        graphs=[];outputs=[]
        for fn in (incumbent,candidate):
            for _ in range(2):fn()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn()
            graphs.append(g);outputs.append(out)
        worst=0.;worst_row=0.
        for repeat in range(6):
            attn.normal_(generator=gen);stream.normal_(generator=gen)
            for g in (graphs if repeat%2==0 else graphs[::-1]):g.replay()
            for got,want in zip(outputs[1],outputs[0]):
                error,row=_best5_error(got,want)
                worst=max(worst,error);worst_row=max(worst_row,row)
        if worst>.01 or worst_row>.05:raise RuntimeError(f'S3 chain gate {key}: {worst:.6f}/{worst_row:.6f}')
        state['best7_gates'][('s3_chain',key)]=dict(fp32=worst,worst_row=worst_row,
            source='actual row projection/glue, six fresh producers, alternating replay order')
