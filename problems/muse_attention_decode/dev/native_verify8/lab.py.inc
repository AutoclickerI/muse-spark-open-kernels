

def lab(deadline,weights):
    import json
    lines=['META native FP8 WGMMA Q-major M64; parallel combine; same-input FA3; no incumbent selection',
           'META clocks: consumer wait/transpose/QK/softmax/P/PV plus producer issue/backpressure; resource and plain/clock wall times']
    if deadline-time.monotonic()<20:return lines+['SKIP native compile budget<20s']
    native_ext(True)
    flush=torch.empty(128<<20,dtype=torch.uint8,device='cuda')
    for B in (16,32):
        if deadline-time.monotonic()<8:lines.append(f'SKIP b{B} budget<8s');break
        data=generate_input(B,80000,'tp8','full',741+B,q_len=8,page=1,spread='agentic',dist='coherent');data['config']['tokens']=B*8
        def control(d):
            q,g=qk_prep((d['qkv'].view(1,-1,2304),1),d,{'inv_freq':None},q_fp8=True)
            return _attention_fa3_public(q,g,d,{})
        us,actual,graph=timed(control,data,flush)
        expected=gate_prepare(data);ok,msg,error=gate_cached(data,actual,expected)
        lines.append(f'FA3 B={B} us={us:.2f} gate={"P" if ok else "F"} err={error:.6f}')
        if not ok:lines.append('GATE_ERROR '+msg)
        logical=float(data['seq_lens'].sum())*256
        for setting in ((9,4),(17,4),(17,8),(33,4)):
            if deadline-time.monotonic()<3:lines.append(f'SKIP b{B} native tail budget<3s');break
            def call(d):return native_attention(d['qkv'],d,setting,force_hopper=True)[0]
            us,out,g=timed(call,data,flush)
            ok,msg,error=gate_cached(data,out,expected)
            trace_out,trace=native_attention(data['qkv'],data,setting,clocks=True,force_hopper=True);torch.cuda.synchronize()
            tok,tmsg,terr=gate_cached(data,trace_out,expected);equal=bool(torch.equal(out,trace_out))
            values=trace.cpu().to(torch.float64);count=values[:,:,9].clamp_min(1)
            consumer=values[:,:4,:7]/count[:,:4,None]
            producer=values[:,4:,:2]/count[:,4:,None]
            cm=consumer.reshape(-1,7).median(0).values.tolist();pm=producer.reshape(-1,2).median(0).values.tolist()
            def trace_call(d):return native_attention(d['qkv'],d,setting,clocks=True,force_hopper=True)[0]
            trace_us,_,trace_graph=timed(trace_call,data,flush,rounds=3)
            resources=[native_ext(True).native_resources(setting[1],i) for i in (0,1)]
            row=dict(B=B,setting=setting,us=round(us,2),tb_s=round(logical/us/1.e6,3),gate=bool(ok and tok and equal),error=round(error,6),trace_equal=equal,
                grid=B*2*setting[0],trace_us=round(trace_us,2),resources=resources,consumer_cycles=[round(x) for x in cm],producer_cycles=[round(x) for x in pm])
            lines.append('NATIVE '+json.dumps(row,separators=(',',':')))
            if not row['gate']:lines.append('GATE_ERROR '+msg+'; '+tmsg)
    return lines
