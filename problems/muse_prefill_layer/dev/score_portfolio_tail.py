# Layouts are resources: every expert contender reads its preferred lossless bank.
_RAW_DG_CHOICES=('raw_v11','raw_dg','raw_dg_hint','raw_dg_w8','raw_dg_hint_w8','raw_dg_k1024_w8','raw_dg_routes')
_EXPERT_CHOICES=tuple(_GROUPED_CHOICES)+('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048','dg_w8','dg_hint_w8','dg_k2048_w8','dg_hint_k2048_w8')+_RAW_DG_CHOICES+('raw11','raw15','raw17','humming','humming_tma','humming_masked','marlin')


def _run_raw_dg(pooled,probs,slot_e,slot_pos,counts,st,name):
    import deep_gemm
    if name=='raw_v11':
        return _raw_experts_dequant(pooled,probs,slot_e,slot_pos,counts,st)
    T,D=pooled.shape;dev=pooled.device;cap=T*8+32*128
    starts=torch.empty(32,dtype=torch.int32,device=dev);psum=torch.empty_like(starts)
    _offsets_kernel[(1,)](counts,starts,psum,ALIGN=128)
    a=torch.empty((cap,D),dtype=torch.bfloat16,device=dev)
    row_of=torch.empty((T,8),dtype=torch.int32,device=dev)
    _gather_norm_kernel[(T,)](pooled,st['pre_expert'],slot_e,slot_pos,starts,a,row_of,D=D,num_warps=8)
    # Both expanded matrices coexist: no artificial workspace objective.
    w13=torch.empty((32,8192,4096),dtype=torch.bfloat16,device=dev)
    w2=torch.empty((32,4096,4096),dtype=torch.bfloat16,device=dev)
    bkb=1024 if 'k1024' in name else 512
    for key,rows,w in (('w13',8192,w13),('w2',4096,w2)):
        _dequant_kernel[(32,rows//8)](st[key],st[key+'_sf'],w,counts,ROWS=rows,K=4096,BR=8,BKB=bkb,num_warps=4)
    h=torch.empty((cap,8192),dtype=torch.bfloat16,device=dev)
    inter=torch.empty_like(a);y=torch.empty_like(a)
    hint={'expected_m_for_psum_layout':max(1,triton.cdiv(2*T,32))} if 'hint' in name else {}
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a,w13,h,psum,use_psum_layout=True,**hint)
    if 'routes' in name:
        _swiglu_routes_kernel[(triton.cdiv(T*8,4),)](h,inter,row_of,slot_e,st['w13_gs'],T,BR=4,D=4096,num_warps=8)
    else:
        _portfolio_swiglu[(32,triton.cdiv(T,4))](h,inter,starts,counts,st['gs13_gate'],st['gs13_up'],D=4096,BR=4,num_warps=8 if 'w8' in name else 4)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(inter,w2,y,psum,use_psum_layout=True,**hint)
    out=torch.empty_like(pooled)
    _combine_kernel[(T,)](y,row_of,slot_e,probs,st['gs2'],st['post_expert_bf16'],out,D=4096,num_warps=8)
    return out


def _score_expert(pooled,probs,slot_e,slot_pos,counts,st,name):
    if name=='marlin':
        xn=torch.empty_like(pooled)
        _pre_expert_norm_kernel[(pooled.shape[0],)](pooled,st['pre_expert'],xn,D=4096,num_warps=4)
        ids=torch.where(slot_e>=0,slot_e,255).int()
        return _experts_marlin(xn,probs,ids,st['zero_start'],st)
    if name in _RAW_DG_CHOICES:
        return _run_raw_dg(pooled,probs,slot_e,slot_pos,counts,st,name)
    if name in ('raw11','raw15','raw17'):
        return _raw_experts_tiled(pooled,probs,slot_e,slot_pos,counts,st['raw_states'][name])
    if name.startswith('humming'):
        hs=dict(st);hs['humming']=st['humming_variants'][name]
        if name=='humming_masked':
            return _experts_humming_masked(pooled,probs,slot_e,slot_pos,counts,hs)
        return _experts_humming(pooled,probs,slot_e,hs)
    if name=='pipeline':return _experts_pipeline(pooled,probs,slot_e,slot_pos,counts,st)
    if name.startswith('dg'):return _run_canonical_dg(pooled,probs,slot_e,slot_pos,counts,st,name)
    xn=torch.empty_like(pooled)
    _pre_expert_norm_kernel[(pooled.shape[0],)](pooled,st['pre_expert'],xn,D=4096,num_warps=4)
    return _run_grouped(xn,probs,slot_e,slot_pos,counts,st,name)


def _experts_selected(pooled,probs,slot_e,slot_pos,counts,st):
    return _score_expert(pooled,probs,slot_e,slot_pos,counts,st,st['expert_dispatch'][_expert_bucket(pooled.shape[0])])


@triton.jit
def _dense_persistent(A,B,C,M,N:tl.constexpr,K:tl.constexpr,
                      BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GRID:tl.constexpr):
    mi=tl.cdiv(M,BM);ni=tl.cdiv(N,BN)
    for tile in range(tl.program_id(0),mi*ni,GRID):
        # Group eight row tiles for weight reuse, including the partial last group.
        group=tile//(8*ni);first=group*8;gs=tl.minimum(mi-first,8)
        within=tile%(8*ni);pm=first+within%gs;pn=within//gs
        rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
        acc=tl.zeros((BM,BN),tl.float32)
        for kg in range(tl.cdiv(K,BK)):
            kk=kg*BK+rk
            av=tl.load(A+rm[:,None].to(tl.int64)*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0)
            bv=tl.load(B+rn[None,:].to(tl.int64)*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0)
            acc+=tl.dot(av,bv)
        tl.store(C+rm[:,None].to(tl.int64)*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))


_DENSE_TRITON={'triton64':(64,128,64,4,3),'triton128':(128,128,64,8,3),'triton256':(128,256,64,8,3)}


def _score_dense_run(x,st,key,choice):
    w=st[key];fp32=key=='router_cat';name=choice[0]
    if name=='cublas':return torch.mm(x,w.t(),**({'out_dtype':torch.float32} if fp32 else {}))
    if name=='cublas_nn':return torch.mm(x,st['dense_transposed'][key],**({'out_dtype':torch.float32} if fp32 else {}))
    out=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=torch.float32 if fp32 else torch.bfloat16)
    if name=='lt':
        codes=st['lt_algorithms'][(key,x.shape[0])]
        st['grouped_mod'].lt_run(w,x,out,st['lt_workspace'],codes,choice[1])
    elif name=='dg':
        import deep_gemm
        deep_gemm.bf16_gemm_nt(x,w,out)
    else:
        bm,bn,bk,warps,stages=_DENSE_TRITON[name]
        _dense_persistent[(st['sms'],)](x,w,out,x.shape[0],N=w.shape[0],K=w.shape[1],BM=bm,BN=bn,BK=bk,
                                      GRID=st['sms'],num_warps=warps,num_stages=stages)
    return out


def _score_dense(x,st,key):
    return _score_dense_run(x,st,key,st['dense_choices'].get((key,x.shape[0]),('cublas',)))


class _ScoreGraph:
    def __init__(self,fn):
        self.stream=torch.cuda.Stream();self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            fn();fn()
        self.stream.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=self.stream):self.output=fn()
        self.graph.replay();self.stream.synchronize()
    def replay(self):
        self.stream.wait_stream(torch.cuda.current_stream());self.graph.replay()
        torch.cuda.current_stream().wait_stream(self.stream)


_SCORE_FIXED_US=0.


def _score_times(graphs,refresh,flush,repeats=3,control=None):
    # Synthetic activations/routing are refreshed before every replay. Flush is outside events.
    # Alternate the order in paired rounds; keep streams separate from cuBLAS graph workspaces.
    values=[[] for _ in graphs]
    for r in range(repeats):
        refresh(r)
        for i in (range(len(graphs)) if r%2==0 else reversed(range(len(graphs)))):
            flush.sum()
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();graphs[i].replay();b.record();b.synchronize()
            values[i].append(max(a.elapsed_time(b)*1000.-_SCORE_FIXED_US,.001))
    import statistics
    if control is None:return [statistics.median(v) for v in values]
    # Paired candidate/control ratios on exactly the same fresh fixture; two rounds
    # reverse launch order. Replay overhead is common and small at these sizes.
    base=values[control]
    return [statistics.median(base)/statistics.median([(base[r]+base[r+1])/(v[r]+v[r+1]) for r in range(0,len(base)-1,2)]) for v in values]


def _score_error(out,ref,limit,where):
    d=out.float()-ref.float();v=float(d.norm()/ref.float().norm().clamp_min(1e-12))
    if not math.isfinite(v) or v>limit:
        raise RuntimeError(f'{where}: relative L2 {v} exceeds {limit}')
    return v


def _time_left(st):
    import time
    return st['_deadline']-time.monotonic()


def _tune_score_dense(st,flush):
    # Exact scored shapes, plus legal generic graph paths for nearby shapes.
    for T in (16384,753,3012):
        for key in ('pooler','unpooler','qkv','o','router_cat'):
            if _time_left(st)<12:
                print('SCORE_TUNING_BUDGET dense remaining shapes use the measured control',flush=True);return
            w=st[key];x=torch.randn((T,w.shape[1]),device=w.device,dtype=torch.bfloat16)
            reference=_ScoreGraph(lambda:_score_dense_run(x,st,key,('cublas',)))
            y=reference.output
            codes=st['grouped_mod'].lt_options(w,x,y)
            st['lt_algorithms'][(key,T)]=codes
            choices=[('cublas',),('cublas_nn',)]
            if key!='router_cat':choices.append(('dg',))
            choices += [(n,) for n in _DENSE_TRITON]
            choices += [('lt',i) for i in range(codes.shape[0])]
            costs={};errors={}
            refresh=lambda r:x.normal_(generator=st['_generator'])
            for choice in choices:
                if _time_left(st)<12:break
                g=reference if choice==('cublas',) else _ScoreGraph(lambda c=choice:_score_dense_run(x,st,key,c))
                reference.replay();torch.cuda.synchronize()
                errors[str(choice)]=_score_error(g.output,reference.output,.0005 if key=='router_cat' else .008,f'dense {key} {T} {choice}')
                costs[choice]=_score_times([g],refresh,flush)[0]
                if g is not reference:del g
            if not costs:costs[('cublas',)]=_score_times([reference],refresh,flush)[0]
            finalists=sorted(costs,key=costs.get)[:3]
            if ('cublas',) not in finalists:finalists.append(('cublas',))
            graphs=[reference if c==('cublas',) else _ScoreGraph(lambda c=c:_score_dense_run(x,st,key,c)) for c in finalists]
            paired=_score_times(graphs,refresh,flush,6,control=finalists.index(('cublas',)))
            selected=finalists[min(range(len(paired)),key=paired.__getitem__)]
            st['dense_choices'][(key,T)]=selected
            st['_measurements']['dense'][str((key,T))]=dict(choices=list(costs),us=[round(v,2) for v in costs.values()],selected=selected)
            print('SCORE_DENSE',json.dumps(dict(tokens=T,projection=key,selected=selected,us={str(k):v for k,v in costs.items()},paired_us=paired,finalists=finalists,relative_l2=errors)),flush=True)
            del graphs,reference,y,x


def _expert_synthetic(T,st,n):
    dev=st['qkv'].device
    x=torch.randn((T,4096),device=dev,dtype=torch.bfloat16,generator=st['_generator'])
    e=torch.full((T,8),-1,dtype=torch.int32,device=dev);p=torch.zeros_like(e)
    probs=torch.zeros((T,8),device=dev)
    r=torch.arange(T*n,device=dev,dtype=torch.int32).reshape(T,n)
    e[:,:n]=r%32;p[:,:n]=r//32;probs[:,:n]=1./n
    counts=(torch.full((32,),T*n//32,device=dev,dtype=torch.int32)+(torch.arange(32,device=dev)<T*n%32).int())
    return x,probs,e,p,counts


def _tune_score_experts(st,flush):
    # Whole expert stage, including gather, layout decode, activation and finalize.
    for T in (753,3012,16384):
        if _time_left(st)<12:
            print('SCORE_TUNING_BUDGET experts remaining shapes use v11 control',flush=True);return
        trials={};errors={}
        data=_expert_synthetic(T,st,2);x=data[0]
        base=_ScoreGraph(lambda:_score_expert(*data,st,'raw11' if T<8192 else 'raw_v11'))
        refresh=lambda r:x.normal_(generator=st['_generator'])
        for name in _EXPERT_CHOICES:
            if _time_left(st)<12:break
            g=_ScoreGraph(lambda n=name:_score_expert(*data,st,n))
            base.replay();torch.cuda.synchronize()
            errors[name]=_score_error(g.output,base.output,.015,f'experts {T} {name}')
            trials[name]=_score_times([g],refresh,flush)[0]
            del g
        # Finalists replay fresh 1- and 2-route synthetic fixtures, paired in swapped order.
        finalists=sorted(trials,key=trials.get)[:3]
        control='raw11' if T<8192 else 'raw_v11'
        if control not in finalists:finalists.append(control)
        del base,data,x
        paired=[]
        for n in (1,2):
            data=_expert_synthetic(T,st,n);x=data[0]
            graphs=[_ScoreGraph(lambda name=name:_score_expert(*data,st,name)) for name in finalists]
            pair=_score_times(graphs,lambda r:x.normal_(generator=st['_generator']),flush,6,control=finalists.index(control))
            paired.append(pair)
            del graphs,data,x
        mean=[math.sqrt(a*b) for a,b in zip(*paired)]
        selected=finalists[min(range(len(mean)),key=mean.__getitem__)]
        st['expert_dispatch'][T]=selected
        st['_measurements']['expert'][T]=dict(choices=list(trials),us=[round(v,2) for v in trials.values()],selected=selected)
        print('SCORE_EXPERT',json.dumps(dict(tokens=T,selected=selected,us=trials,finalists=finalists,paired_us=paired,relative_l2=errors)),flush=True)


_SCORE_ATTENTION_CHOICES=('fa3','fa3_contiguous','fa3_unpacked','fp8_direct128','fp8_gather128','fp8_gather256')


def _score_attention_path(data,q8,st,name):
    if name.startswith('fa3_'):
        return _attention_contiguous(data,q8,st,unpacked=name=='fa3_unpacked')
    return _attention_path(data,q8,st,name)


def _score_attention_key(data):
    cfg=data['config']
    return cfg['tokens'],cfg['window'],data['page_table'].shape[1]*cfg['page_size']-cfg['q_len']


def _attention(data,q8,st):
    return _score_attention_path(data,q8,st,st['attention_choices'].get(_score_attention_key(data),'fa3'))


def _tune_score_attention(st,flush):
    cases=((1,753,80000,2048),(1,753,80000,0),(4,753,80000,2048),(4,753,80000,0),
           (1,16384,0,0),(1,16384,32768,2048),(1,16384,32768,0),(1,16384,65536,0))
    for B,L,prefix,window in cases:
        if _time_left(st)<12:
            print('SCORE_TUNING_BUDGET attention remaining shapes use FA3 control',flush=True);return
        S=prefix+L;dev=st['qkv'].device;gen=st['_generator'];T=B*L
        q8=torch.empty((T,1024),device=dev,dtype=torch.float8_e4m3fn)
        cache=torch.empty((B*S,1,2,64),device=dev,dtype=torch.float8_e4m3fn)
        data=dict(config=dict(tokens=T,batch=B,q_len=L,window=window,page_size=1,sm_scale=1/8,qk_scale_factor=5.4730064863875,kv_scale=1.),
                  k_cache=cache,v_cache=torch.empty_like(cache),
                  page_table=torch.randperm(B*S,device=dev,generator=gen,dtype=torch.int64).int().view(B,S),
                  seq_lens=torch.full((B,),prefix,device=dev,dtype=torch.int32),cache_seqlens=torch.full((B,),S,device=dev,dtype=torch.int32),
                  cu_seqlens_q=torch.arange(B+1,device=dev,dtype=torch.int32)*L,scheduler_metadata=None,
                  k_scale=torch.ones(1,device=dev),v_scale=torch.ones(1,device=dev))
        def refresh(r):
            q=torch.randn((T,16,64),device=dev,generator=gen);q*=torch.rsqrt(q.square().mean(-1,keepdim=True));q8.copy_(q.reshape(T,1024))
            k=torch.randn(cache.shape,device=dev,generator=gen);u=torch.randn((2,64),device=dev,generator=gen)
            u*=torch.rsqrt(u.square().mean(-1,keepdim=True));k.mul_(.44**.5).add_(u,alpha=.56**.5)
            cache.copy_(k);data['v_cache'].copy_(torch.randn(cache.shape,device=dev,generator=gen))
        refresh(0);base=_ScoreGraph(lambda:_score_attention_path(data,q8,st,'fa3'))
        graphs=[base];names=['fa3'];errors={'fa3':0.}
        for name in _SCORE_ATTENTION_CHOICES[1:]:
            if _time_left(st)<8:break
            g=_ScoreGraph(lambda n=name:_score_attention_path(data,q8,st,n))
            errors[name]=_score_error(g.output,base.output,.04,f'attention {B} {L} {prefix} {window} {name}')
            graphs.append(g);names.append(name)
        times=_score_times(graphs,refresh,flush,6,control=0)
        chosen=names[min(range(len(times)),key=times.__getitem__)]
        key=_score_attention_key(data);st['attention_choices'][key]=chosen
        st['_measurements']['attention'][str(key)]=dict(names=names,us=[round(t,2) for t in times],selected=chosen)
        print('SCORE_ATTENTION',json.dumps(dict(shape=key,selected=chosen,us=dict(zip(names,times)),relative_l2=errors)),flush=True)
        del base,graphs,data,q8,cache


class _RawModule:
    def __init__(self,mod,version):self.moe_gemm=getattr(mod,version+'_gemm')


def prepare(weights):
    global _SCORE_FIXED_US
    import time,copy
    started=time.monotonic()
    import deep_gemm
    deep_gemm.use_deterministic_algorithms(True)
    st=prepare_base(weights)
    st['_deadline']=started+280
    st['_generator']=torch.Generator(device=st['qkv'].device).manual_seed(478191)
    st['grouped_mod']=_grouped_module();st['tiled_mod']=st['grouped_mod']
    st['sms']=torch.cuda.get_device_properties(st['qkv'].device).multi_processor_count
    st['gs13_gate']=weights.get('gs13_gate',weights['w13_gs']).float().contiguous()
    st['gs13_up']=weights.get('gs13_up',weights['w13_gs']).float().contiguous()
    st['gs2']=weights['w2_gs'].float().contiguous();st['post_expert']=st['post_expert_bf16']
    st['w13p']=repack_v6(weights['w13'],weights['w13_sf'],True);st['w2p']=repack_v6(weights['w2'],weights['w2_sf'],False)
    raw=dict(st)
    raw['blob13'],raw['sfb13']=_raw_repack(weights['w13'],weights['w13_sf'],True)
    raw['blob2'],raw['sfb2']=_raw_repack(weights['w2'],weights['w2_sf'],False)
    st['raw_states']={}
    for name in ('raw11','raw15','raw17'):
        sub=dict(raw);sub['tiled_mod']=_RawModule(st['grouped_mod'],name)
        if name=='raw17':
            # Reuse the already-repacked raw bank, then encode bit planes losslessly.
            for key,sfkey in (('blob13','sfb13'),('blob2','sfb2')):
                b=raw[key];s=raw[sfkey];o=torch.empty_like(b);so=torch.empty_like(s)
                _encode_blob_fast[(b.numel()//2048,)](b.view(torch.int32),s,o.view(torch.int32),num_warps=4)
                _encode_sf_fast[(triton.cdiv(s.numel(),1024),)](s,so,s.numel(),BLOCK=1024,num_warps=4)
                sub[key]=o;sub[sfkey]=so
        st['raw_states'][name]=sub
    hum=_Experts(weights,0,32)
    st['humming_variants']={'humming':hum}
    for name,tma,gemm_type in (('humming_tma',True,GemmType.INDEXED),('humming_masked',False,GemmType.GROUPED_MASKED)):
        h=copy.copy(hum);h.tma=tma;h.gemm_type=gemm_type;h.kernels={'w13':{},'w2':{}};h.plans={}
        h.compile([1<<i for i in range(16)]);st['humming_variants'][name]=h
    marlin_weights={k:(v.clone() if k.startswith(('w13','w2')) else v) for k,v in weights.items()}
    st['bank']=_marlin_bank(marlin_weights)
    st['zero_start']=torch.zeros((),device=st['qkv'].device,dtype=torch.int32)
    st['ones']=torch.ones(32,device=st['qkv'].device)
    st['dense_transposed']={key:st[key].t().contiguous() for key in ('qkv','o','pooler','unpooler','router_cat')}
    st['lt_algorithms']={};st['lt_workspace']=torch.empty(256*1024*1024,device=st['qkv'].device,dtype=torch.uint8)
    st['dense_choices']={};st['expert_dispatch']={753:'raw11',3012:'raw11',16384:'raw_v11'}
    st['attention_choices']={}
    st['_measurements']={'dense':{},'expert':{},'attention':{}}
    if IS_HOPPER:
        # Dense first: its library/kernel candidates compile quickly and are most under-explored.
        # Every omitted trial is reported if the external 300-second limit approaches.
        flush=torch.zeros(1<<30,device=st['qkv'].device,dtype=torch.uint8)
        replay_token=torch.zeros(1,device=st['qkv'].device)
        replay_graph=_ScoreGraph(lambda:replay_token.add_(1))
        _SCORE_FIXED_US=0.
        _SCORE_FIXED_US=_score_times([replay_graph],lambda r:None,flush,16)[0]
        print('SCORE_GRAPH_FIXED_US',_SCORE_FIXED_US,flush=True)
        del replay_graph,replay_token
        _tune_score_dense(st,flush)
        _tune_score_experts(st,flush)
        _tune_score_attention(st,flush)
        del flush
    torch.cuda.synchronize()
    st.pop('_generator');st.pop('_deadline')
    print('SCORE_COMPLETE_TABLE',json.dumps(st.pop('_measurements'),separators=(',',':')),flush=True)
    print('SCORE_PREPARE',json.dumps(dict(seconds=time.monotonic()-started,expert=st['expert_dispatch'],dense={str(k):v for k,v in st['dense_choices'].items()},attention={str(k):v for k,v in st['attention_choices'].items()},expert_contenders=len(_EXPERT_CHOICES),graph_fixed_us=_SCORE_FIXED_US)),flush=True)
    return st
