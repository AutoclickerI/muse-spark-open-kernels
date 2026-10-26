def dl_namespace():
    import linecache
    file='<p10-decoder-source>';ns={'__name__':'p10_decoder'}
    linecache.cache[file]=(len(DC_SOURCE),None,DC_SOURCE.splitlines(True),file);exec(compile(DC_SOURCE,file,'exec'),ns)
    return ns


def lab(deadline,weights,state=None):
    import math,time,statistics,hashlib
    from concurrent.futures import ThreadPoolExecutor
    from torch.utils.cpp_extension import load_inline
    begin=time.monotonic();m=dl_namespace();fast={};exec(FAST_SOURCE,fast);T=torch
    g=T.Generator(device='cuda').manual_seed(61791)
    def dense(n,k):return (T.randn((n,k),device='cuda',generator=g)/math.sqrt(k)).bfloat16()
    if weights is None or 'pooler' not in weights:
        weights=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),unpooler=dense(8192,4096))
        mean=T.randn(8192,generator=T.Generator().manual_seed(5354)).cuda()
        weights['router']=.045*T.randn((256,8192),device='cuda',generator=g)+.07*T.randn((256,1),device='cuda',generator=g)*mean/math.sqrt(8192)
    roles=m['dc_role_weights'](weights);native=T.cuda.get_device_capability()==(9,0)
    keys=set(m['NF_PORT_SELECTED'])-{(32,'o')}
    m['cd_module'](((1,1,2,3),));ctrl=dict(roles);m['_seed_v9_gemv_choices'](ctrl)
    ctrl['pre_expert_w']=T.ones(4096,device='cuda');ctrl['router_bias']=T.zeros(256,device='cuda')
    scalar_native=m['DC_NATIVE'];vector_native=fast['vector_source'](scalar_native);view_native=fast['view_source'](scalar_native)
    def prepare_variant(which):
        # Each builder owns its globals; parallel JIT must not change another
        # arm's reader or load_inline name while that arm is being instantiated.
        ns=dl_namespace();ns['DC_NATIVE']=(scalar_native,vector_native,view_native)[which]
        if which==2:fast['install_view'](ns)
        return ns['dc_nf_prepare'](weights,keys),ns['dc_cd_module']()
    with ThreadPoolExecutor(4) as pool:
        future_raw=pool.submit(m['_dc_base_nf_prepare'],weights,keys)
        futures=[pool.submit(prepare_variant,i) for i in range(3)]
        m['_cd_prepare'](ctrl)
        banks={n:m['dc_encode_cd'](ctrl['coded_dense'][n])[0] for n in ('qkv_gate','pool_router')}
        raw_nf=future_raw.result();(scalar_nf,scalar_cd),(vector_nf,vector_cd),(view_nf,view_cd)=[f.result() for f in futures]
    T.cuda.synchronize()
    for key,p in scalar_nf['plans'].items():
        for bank in (vector_nf,view_nf):assert T.equal(p['wp'],bank['plans'][key]['wp']) and T.equal(p['esc'],bank['plans'][key]['esc']),('changed bank',key)
    src=SCAN_SOURCE.replace('DC_SCALAR',scalar_native.replace('dc_','ds_')).replace('DC_VECTOR',vector_native.replace('dc_','dv_')).replace('DC_VIEW',view_native.replace('dc_','dh_'))
    arch='90a' if native else ''.join(map(str,T.cuda.get_device_capability()))
    tag=hashlib.sha256(src.encode()).hexdigest()[:12]
    scanner=load_inline(name='p10_scan_'+tag+'_'+arch,cpp_sources='void scan(uint64_t,uint64_t,uint64_t,int,int,uint64_t);int resources(int,int);',cuda_sources=src,functions=['scan','resources'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    def capture(fn):
        fn();T.cuda.synchronize();graph=T.cuda.CUDAGraph()
        with T.cuda.graph(graph):out=fn()
        graph.replay();T.cuda.synchronize();return graph,out
    def times(graphs,n=8):
        ev=[]
        for r in range(n):
            order=range(len(graphs)) if r%2==0 else range(len(graphs)-1,-1,-1)
            for j in order:
                lab_cold_l2();a,b=T.cuda.Event(enable_timing=True),T.cuda.Event(enable_timing=True)
                a.record();graphs[j].replay();b.record();ev.append((j,a,b))
        T.cuda.synchronize();vals=[[] for _ in graphs]
        for j,a,b in ev:vals[j].append(a.elapsed_time(b)*1000)
        return [statistics.median(v) for v in vals]
    empty=times([capture(lambda:None)[0]])[0]
    lines=[f'META same-bank scalar/vector/hoisted-view; S15 A/A2;8seeds8cold orders; setup={time.monotonic()-begin:.1f}s',
        'META scan includes nonlinear per-word hash, same in all arms (not a pure copy roof); opaque skips decode; view scan rebuilds context per unit, stage hoists it']
    lines.append('REGS '+str([(i,scanner.resources(i,0),scanner.resources(i,1),scanner.resources(i,2)) for i in range(5)]))
    def xor_rows(v):
        pad=T.zeros((v.shape[0],m['triton'].next_power_of_2(v.shape[1])),device=v.device,dtype=T.int32);pad[:,:v.shape[1]]=v
        while pad.shape[1]>1:pad=pad[:,:pad.shape[1]//2]^pad[:,pad.shape[1]//2:]
        return pad[:,0]
    def reader_hash(words):
        v=words.reshape(-1,3,32,4).permute(0,2,1,3).reshape(-1,32,12).long()&0xffffffff
        total=T.zeros(v.shape[:2],device=v.device,dtype=T.int32)
        for i in range(12):
            z=(v[:,:,i]*((1664525+1013904223*(i+1))&0xffffffff))&0xffffffff;shift=(i*7)&31
            z=((z>>shift)|(z<<(32-shift)))&0xffffffff
            total^=z.int()
        return xor_rows(total)
    done=0
    for t,name in sorted(keys|{(1,'qkv_gate'),(1,'pool_router')}):
        if time.monotonic()>deadline-15:return lines+[f'INCOMPLETE {done}/10; no promotion']
        w=roles[name];x=T.randn((t,w.shape[1]),device='cuda',dtype=T.bfloat16)
        data={'local_start':T.zeros((),device='cuda',dtype=T.int32)}
        def produce(which):
            if (t,name) in m['CD_ROWS']:
                if which==0:return m['_nf_port_base_project'](x,w,name,ctrl,'custom',T.float32)
                m['_DC_CD_MODULE']=(scalar_cd,vector_cd,view_cd)[which-1]
                return m['dc_cd_gemv'](x,banks[name],m['CD_ROWS'][t,name])
            bank=(raw_nf,scalar_nf,vector_nf,view_nf)[which]
            if not native and (t,name) not in bank['plans']:
                return T.mm(x,w.T,out_dtype=T.float32).unsqueeze(0),1
            return m['_dc_base_nf_project'](x,name,bank)
        def consume(which):
            pair=produce(which)
            if name=='pool_router':
                return m['_glue_cl_route'](pair,ctrl,data,('cl',8,256,0)) if native else m['route_and_norm'](pair,ctrl,data)
            return m['_materialize'](pair)
        saved=[capture(lambda:consume(0)),capture(lambda:consume(0)),capture(lambda:consume(1)),capture(lambda:consume(2)),capture(lambda:consume(3))]
        vals=[[] for _ in saved];errors=[0.,0.,0.];probes=[0.,0.,0.]
        for seed in (61791,14357,92003,17453,51613,72577,35111,81901):
            g.manual_seed(seed);x.copy_(T.randn(x.shape,device='cuda',generator=g).bfloat16())
            ref=x.double()@w.double().T
            control=m['_materialize'](produce(0)).double();control_error=float((control-ref).norm()/ref.norm());assert control_error<3e-5,('CONTROL',t,name,control_error)
            for j in (1,2,3):
                pair=produce(j);got=m['_materialize'](pair).double();e=float((got-ref).norm()/ref.norm());assert e<3e-5,(t,name,j,e);errors[j-1]=max(errors[j-1],e)
            for graph,_ in saved:graph.replay();T.cuda.synchronize()
            if name=='pool_router':
                a=saved[0][1]
                for _,b in saved[2:]:assert T.equal(a[1],b[1]) and T.equal(a[3],b[3]) and float((a[2]-b[2]).norm()/a[2].norm())<1e-4,(name,t,'route')
            original=x.clone();x.zero_();x[:,(seed*131+t)%x.shape[1]]=16;pr=x.double()@w.double().T
            control=m['_materialize'](produce(0)).double();control_probe=float((control-pr).norm()/pr.norm());assert control_probe<1e-5,('CONTROL probe',t,name,control_probe)
            for j in (1,2,3):
                got=m['_materialize'](produce(j)).double();e=float((got-pr).norm()/pr.norm());assert e<1e-5,(t,name,j,'probe',e);probes[j-1]=max(probes[j-1],e)
            x.copy_(original);got=times([a[0] for a in saved])
            for j,v in enumerate(got):vals[j].append(v-empty)
        aa=max(abs(b/a-1) for a,b in zip(vals[0],vals[1]));bs=[b/a for a,b in zip(vals[0],vals[2])];vs=[b/a for a,b in zip(vals[0],vals[3])];hs=[b/a for a,b in zip(vals[0],vals[4])]
        lines.append(f'R {t}:{name} A={statistics.median(vals[0]):.2f} scalar={statistics.median(vals[2]):.2f} vector={statistics.median(vals[3]):.2f} view={statistics.median(vals[4]):.2f} rS={statistics.median(bs):.4f} rV={statistics.median(vs):.4f} rH={statistics.median(hs):.4f}[{min(hs):.4f},{max(hs):.4f}] aa={aa:.4f} e={max(errors):.1e} p={max(probes):.1e}')
        if (t,name) in m['CD_ROWS']:
            raw=ctrl['coded_dense'][name]['coded'];code=banks[name]['coded']
        elif (t,name) in scalar_nf['plans']:
            raw=raw_nf['plans'][t,name]['wp'];code=scalar_nf['plans'][t,name]['wp']
        else:
            c=m['cd_encode'](w);raw=c['coded'];code=m['dc_encode_cd'](c)[0]['coded']
        units=raw.numel()//1536;refbits=reader_hash(raw.view(T.int32))
        scan_out=[T.empty(units,device='cuda',dtype=T.int32) for _ in range(6)]
        def scan(j):
            scanner.scan(raw.data_ptr(),code.data_ptr(),scan_out[j].data_ptr(),units,(0,0,1,2,3,4)[j],T.cuda.current_stream().cuda_stream);return scan_out[j]
        scans=[capture(lambda j=j:scan(j))[0] for j in range(6)]
        for out in scan_out[:5]:assert T.equal(out,refbits),(name,t,'reader checksum')
        hdr=code[:128].view(T.int32).tolist();body=code[hdr[6]:hdr[6]+units*1408].view(T.int32).reshape(units,352)
        meta=code[hdr[5]:hdr[5]+units*8].view(T.int32).long().reshape(units,2);present=meta[:,0]&0xffffffff;off=meta[:,1]
        lane=T.arange(32,device='cuda');bits=(present[:,None]>>lane)&1;rank=bits.cumsum(1)-bits
        count=int(bits.sum());extensions=code[hdr[7]:hdr[7]+count*4].view(T.int32);extensions=T.cat((extensions,T.zeros(1,device='cuda',dtype=T.int32)))
        values=extensions[T.where(bits.bool(),off[:,None]+rank,count)]
        codes=T.cat((body[:,256:].reshape(units,3,32).permute(0,2,1),values[:,:,None]),2)
        opaque=reader_hash(T.cat((body[:,:256].reshape(units,2,32,4),codes[:,None]),1))
        assert T.equal(scan_out[5],opaque),(name,t,'opaque checksum')
        stimes=[v-empty for v in times(scans,16)];saa=abs(stimes[1]/stimes[0]-1)
        lines.append(f'SCAN {t}:{name} raw={stimes[0]:.2f} scalar={stimes[2]:.2f} vector={stimes[3]:.2f} view={stimes[4]:.2f} opaque={stimes[5]:.2f} aa={saa:.4f} rawTB={raw.numel()/stimes[0]/1e6:.3f} denseW={w.numel()}')
        done+=1
    return lines+[f'END complete={done}/10 elapsed={time.monotonic()-begin:.1f}s']
