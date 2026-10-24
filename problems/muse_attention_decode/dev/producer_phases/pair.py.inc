

def layer_fa3(qkv,data,state,glue='current'):
    # Both current and best5 glue converge on the exact public FA3 call.
    q,gate=(qk_prep if glue=='current' else _best7_old_qk_prep)(qkv,data,state,q_fp8=True)
    return _attention_fa3_public(q,gate,data,state)


def lab(deadline,weights):
    lines=['META verify1 sliding: exact guarded adapter vs decode public FA3 incl qk_prep and norm_gate; no projection GEMV',
           'META q_len=8 B=1 ctx=80000 window2048 page1; PDL=True control; BF16 and FP32 partial ABIs']
    if deadline-time.monotonic()<25:return lines+['SKIP budget<25s before compile']
    _window_ext()
    data=generate_input(1,80000,'tp8','sliding',931,q_len=8,page=1,dist='soft')
    data['config']['tokens']=8
    theta=data['config']['rope_theta']
    state={'inv_freq':1./(theta**(torch.arange(0,64,2,device='cuda').float()/64.))}
    flush=torch.empty(128<<20,dtype=torch.uint8,device='cuda')
    for count in (0,1,4):
        if deadline-time.monotonic()<8:lines.append(f'SKIP partials{count} budget<8s');break
        raw=data['qkv'].view(8,2304)
        if count:
            raw=torch.randn((count,8,2304),device='cuda',dtype=torch.float32)
            data['qkv']=raw.sum(0).to(torch.bfloat16).view(1,8,2304)
            qkv=(raw,count)
        else:qkv=(raw.unsqueeze(0),1)
        first=attention_verify1_sliding(qkv,data);torch.cuda.synchronize()
        expected=gate_prepare(data)
        # Fresh values arrive through the same captured projection-buffer ABI.
        for name in ('single','fa3_current','fa3_best5'):
            if deadline-time.monotonic()<3:lines.append('SKIP paired tail budget<3s');break
            def call(d):
                if name=='single':return attention_verify1_sliding(qkv,d)
                return layer_fa3(qkv,d,state,glue='current' if name=='fa3_current' else 'best5')
            us,out,g=timed(call,data,flush,rounds=7)
            expected=gate_prepare(data)
            ok,msg,err=gate_cached(data,out,expected)
            # Modify values, then verify the captured graph used them.
            saved=raw.clone()
            raw.add_(.015625)
            data['qkv']=(raw.sum(0) if count else raw).to(torch.bfloat16).view(1,8,2304)
            g.replay();torch.cuda.synchronize();fresh=gate_prepare(data)
            fok,fmsg,ferr=gate_cached(data,out,fresh)
            lines.append(f'ROW partials={count} kernel={name} us={us:.2f} gate={"P" if ok and fok else "F"} err={err:.6f} fresh_err={ferr:.6f} graph=P')
            if not(ok and fok):lines.append('GATE_ERROR '+msg+'; '+fmsg)
            raw.copy_(saved)
            data['qkv']=(raw.sum(0) if count else raw).to(torch.bfloat16).view(1,8,2304)
    return lines
