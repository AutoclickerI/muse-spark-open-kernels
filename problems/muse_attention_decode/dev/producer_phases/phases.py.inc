
_EXT_PHASE={}
PHASE_NAMES=('issue','wait','transpose','qk','softmax','p_convert','pv_layout')

def phase_ext(wide=False):
    global _EXT_PHASE
    if wide not in _EXT_PHASE:
        import hashlib
        major,minor=torch.cuda.get_device_capability();arch=f'{major}{minor}'
        source=WIDE_CUDA if wide else PHASE_CUDA
        _EXT_PHASE[wide]=load_inline(name=f'verify8_phase_{hashlib.sha256(source.encode()).hexdigest()[:12]}_{arch}',
            cpp_sources=PHASE_CPP,cuda_sources=source,functions=['verify8_forward','verify8_resources'],
            extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'],verbose=False)
    return _EXT_PHASE[wide]


def phase_run(data,setting,profile,wide=False):
    cfg=data['config'];B=cfg['batch'];HQ=cfg['num_q_heads'];HK=cfg['num_kv_heads'];ns,st,kt=setting
    parts=torch.empty((B*HK*ns*64*68,),device='cuda',dtype=torch.float32)
    out=torch.empty((B,8,1024),device='cuda',dtype=torch.bfloat16)
    clocks=torch.empty((B*HK*ns,4,10),device='cuda',dtype=torch.uint64) if profile else torch.empty(1,device='cuda',dtype=torch.uint64)
    phase_ext(wide).verify8_forward(data['qkv'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),
        data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),parts.data_ptr(),out.data_ptr(),
        B,HQ,HK,data['qkv'].shape[-1],data['page_table'].shape[1],1,ns,0,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,st,kt,int(profile),clocks.data_ptr(),
        torch.cuda.current_stream().cuda_stream)
    return out,clocks



def phase_fa3(data):
    raw=data['qkv'].view(1,-1,2304)
    q,g=qk_prep((raw,1),data,{'inv_freq':None},q_fp8=True)
    return _attention_fa3_public(q,g,data,{})


def lab(deadline,weights):
    import json
    lines=['META measured be19da94 producer; synthetic task inputs; clock cycles per stage; PV includes P layout shuffles']
    if deadline-time.monotonic()<25:return lines+['SKIP budget<25s before compile']
    ext=phase_ext();phase_ext(True);sms=torch.cuda.get_device_properties('cuda').multi_processor_count
    lines.append(f'META sms={sms} clocks instrument every warp; compare plain/profile overhead and resource limits')
    flush=torch.empty(128<<20,dtype=torch.uint8,device='cuda')
    for B in (16,32):
        if deadline-time.monotonic()<12:lines.append(f'SKIP b{B} budget<12s');break
        data=generate_input(B,80000,'tp8','full',741+B,q_len=8,page=1,spread='agentic',dist='coherent')
        data['config']['tokens']=B*8
        control_us,control,control_graph=timed(phase_fa3,data,flush)
        control_expected=gate_prepare(data);cok,cmsg,cerr=gate_cached(data,control,control_expected)
        lines.append(f'FA3 B={B} us={control_us:.2f} gate={"P" if cok else "F"} err={cerr:.6f}')
        if not cok:lines.append('GATE_ERROR '+cmsg)
        base,_=phase_run(data,(17,3,128),False);torch.cuda.synchronize()
        expected=gate_prepare(data);logical=float(data['seq_lens'].sum())*2*2*64
        for wide,setting in ((False,(17,3,128)),(True,(17,3,128)),(True,(34,3,64)),(False,(17,4,128))):
            if deadline-time.monotonic()<4:lines.append(f'SKIP b{B} remaining configs budget<4s');break
            ns,st,kt=setting
            def plain(d):return phase_run(d,setting,False,wide)[0]
            us,out,g=timed(plain,data,flush)
            ok,msg,err=gate_cached(data,out,expected)
            traced,clocks=phase_run(data,setting,True,wide);torch.cuda.synchronize()
            equal=bool(torch.equal(out,traced));tok,tmsg,terr=gate_cached(data,traced,expected)
            def traced_fn(d):return phase_run(d,setting,True,wide)[0]
            instrument_us,_,tg=timed(traced_fn,data,flush,rounds=3)
            values=clocks.cpu().to(torch.float64).reshape(-1,10);counts=values[:,9].clamp_min(1)
            samples=values[:,:7]/counts[:,None]
            med=samples.median(0).values.tolist();p90=torch.quantile(samples,.9,dim=0).tolist()
            resources=[phase_ext(wide).verify8_resources(st,kt,i) for i in (0,1)]
            longest=int(data['seq_lens'].max());chunk=((longest+ns-1)//ns+kt-1)//kt*kt
            row=dict(B=B,wide=wide,setting=setting,us=round(us,2),tb_s=round(logical/us/1.e6,3),gate=bool(ok and tok and equal),
                error=round(err,6),trace_equal=equal,trace_us=round(instrument_us,2),grid=B*2*ns,
                cta_kv_max_bytes=chunk*128,resources=resources,cta_cycles=[round(values[:,7].median().item()),round(torch.quantile(values[:,7],.9).item()),round(values[:,8].median().item()),round(torch.quantile(values[:,8],.9).item())],
                cycles={k:[round(a),round(b)] for k,a,b in zip(PHASE_NAMES,med,p90)})
            lines.append('ROW '+json.dumps(row,separators=(',',':')))
            if not row['gate']:lines.append('GATE_ERROR '+msg+'; '+tmsg)
        # Graphs and closures leave scope on the next iteration; no captured name is deleted.
    return lines
