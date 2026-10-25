def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if isinstance(v,torch.Tensor))
    host=types.ModuleType('full128_diagnostic_s11');fname='<full128_diagnostic_s11>'
    linecache.cache[fname]=(len(TL_S11),None,TL_S11.splitlines(True),fname);exec(compile(TL_S11,fname,'exec'),host.__dict__)
    if torch.cuda.get_device_capability()[0]!=9:host._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
    src=PF_CUDA+'\n'+D128_TILE;tag=hashlib.sha256(src.encode()).hexdigest()[:12]
    tile=load_inline(name=f'full128_math_{tag}_{arch}',cpp_sources='#include <cstdint>\nvoid full128_tile_forward(int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t);',cuda_sources=src,
        functions=['full128_tile_forward'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    lines=['META N128 correctness diagnostic: native tile QK/softmax/PV vs independent Torch, then stage3/4 and fully waited controls over context boundaries.',
        'META synthetic inputs only; no performance claim; nonfinite/gate failures are reported per case, never accepted.']
    for masked in (0,1):
        q=torch.randn((2,64,64),device=dev).to(torch.float8_e4m3fn);k=torch.randn((2,128,64),device=dev).to(torch.float8_e4m3fn);v=torch.randn_like(k,dtype=torch.float32).to(k.dtype);out=torch.empty((2,64,64),device=dev)
        tile.full128_tile_forward(q.data_ptr(),k.data_ptr(),v.data_ptr(),out.data_ptr(),.125*1.4426950408889634,masked,torch.cuda.current_stream().cuda_stream)
        logits=q.float()@k.float().transpose(-1,-2)*.125
        if masked:logits.masked_fill_(torch.arange(128,device=dev)[None,None,:]>=96+torch.arange(64,device=dev)[None,:,None]//8,-torch.inf)
        want=logits.softmax(-1)@v.float();torch.cuda.synchronize()
        lines.append('TILE '+str(masked)+' nonfinite='+str(int((~torch.isfinite(out)).sum()))+' rel='+str(float((out-want).norm()/want.norm())))
    native=pf_module(3)
    serial_src=PF_CUDA.replace('  wg_wait<1>();','  wg_wait<0>();')
    tag=hashlib.sha256(serial_src.encode()).hexdigest()[:12]
    serial=load_inline(name=f'full128_serial_{tag}_{arch}',cpp_sources=PF_CPP,cuda_sources=serial_src,functions=['vq8_forward','vq8_resources'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}']) if 'D128_LOCAL_STANDIN' not in globals() else native
    for B,ctx in ((1,20),(1,127),(1,128),(1,257),(1,1000),(1,80000),(16,80000)):
        if time.monotonic()>deadline-4:lines.append('INCOMPLETE '+str((B,ctx)));break
        qkv,data=host._attn_port_synthetic(B,8,0,dev,ctx)
        for name,mod in (('native',native),('serial',serial)):
            PF_MOD[False]=mod
            for st in (3,4):
                got=pf_forward(qkv,data,st);torch.cuda.synchronize()
                bad=int((~torch.isfinite(got)).sum());stats={'b':B,'ctx':ctx,'kind':name,'st':st,'nonfinite':bad}
                try:
                    host._attention_store_agrees(qkv,data)
                    if not bad:stats['gate']=host._attention_agrees(got,host._attention_reference(qkv,data),host._attention_reference(qkv,data,True))
                except Exception as exc:stats['cache_error']=str(exc)
                lines.append('CASE '+json.dumps(stats,separators=(',',':')))
    PF_MOD[False]=native
    return lines
