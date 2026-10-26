SEEDS=tuple(31000033+104729*i for i in range(12))
CASES=((16,1),(32,1),(16,8),(32,8))
ROUNDS=8
MODES={'O':0,'O2':0,'M':0,'M2':0,'C':1,'S':3,'X':7,'N':4,'L':13,'Z':29,'E':32,'W':64,'G':0}
VALID=('O','O2','M','M2','C','W')
RESULTS=[]

def oracle_profile(act,g,legend):
    lab_cold_l2();torch.cuda.synchronize();act.profile_begin();g.replay();torch.cuda.synchronize();items=act.profile_end()
    if not items:return []
    origin=min(v[0] for _,v in items);rows=[]
    for name,v in items:
        if name not in legend:legend.append(name)
        rows.append([legend.index(name),v[0]-origin,v[1]-origin,*v[7:13]])
    return rows

def oracle_clocks(data,coords,family,mode):
    B=data['config']['batch'];buf=torch.zeros((B*2*16,24),device='cuda',dtype=torch.int64)
    name=family+'CLOCK';attention(data,coords,name,mode,buf);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=attention(data,coords,name,mode,buf)
    for _ in range(2):buf.zero_();lab_cold_l2();g.replay();torch.cuda.synchronize()
    rows=[r for r in buf.cpu().tolist() if r[0]];origin=min(r[0] for r in rows)
    summary=lambda x:[min(x),statistics.median(x),max(x)] if x else None
    phases={str(i):summary([r[i]-origin for r in rows if r[i]]) for i in range(9)}
    deltas={f'{i}->{j}':summary([r[j]-r[i] for r in rows if r[i] and r[j]]) for i,j in zip(range(8),range(1,9))}
    cycles={str(i):summary([r[i] for r in rows]) for i in range(10,17)}
    return dict(mode=mode,groups=len(rows),phase_ns=phases,deltas_ns=deltas,cycles=cycles,note='min/median/max; diagnostic clocks alter code. Ready-input PDL protocol cost, not active producer wait.')

def lab(deadline,weights,state=None):
    global RESULTS
    started=time.monotonic();RESULTS=[];spec={};exec(SPEC,spec);host=diag_host();act=diag_activity();legend=[]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(module,SOURCES))
    for B,L in CASES:
        family='d' if L==1 else 'v';names=list(MODES)+(['F'] if L==8 else [])
        times={n:[] for n in names};gates={n:0 for n in names if n in VALID or n=='F'};profiles={};clocks=[]
        resources={k:module(family+k).sliding_resources(L,1,128,1 if LOCAL and family=='d' else (8 if B==16 else 4) if family=='d' else 2) for k in ('A','M','CLOCK')}
        for si,seed in enumerate(SEEDS):
            if time.monotonic()>deadline-15:raise TimeoutError(('incomplete oracle',B,L,si))
            d=spec['generate_input'](B,80000,'tp8','sliding',seed,q_len=L,spread='agentic',dist='soft',page=1);d['config']['tokens']=B*L
            coords=coordinates(d);graphs={};outs={};qkv=(d['qkv'].reshape(1,-1,host.QKV_SIZE),1)
            rotated=names[si%len(names):]+names[:si%len(names)]
            for n in rotated:
                if n=='G':
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):pass
                    graphs[n]=g;continue
                def call(n=n):
                    if n=='F':return diag_call(host,qkv,d,0)
                    return attention(d,coords,family+('A' if n in ('O','O2') else 'M'),MODES[n])
                for _ in range(2):call()
                torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):outs[n]=call()
                graphs[n]=g
            checker=diag_checker(spec,d)
            for n in gates:
                graphs[n].replay();torch.cuda.synchronize();ok,msg=checker(outs[n]);assert ok,(B,L,seed,n,msg);gates[n]+=1
            del checker
            per={n:[] for n in names}
            for ri in range(ROUNDS):
                order=names[ri%len(names):]+names[:ri%len(names)]
                if ri&1:order=order[::-1]
                for n in order:
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);lab_cold_l2();a.record();graphs[n].replay();b.record();b.synchronize();per[n].append(a.elapsed_time(b)*1e6)
            for n in names:times[n].append(round(statistics.mean(per[n])))
            if si==0:
                profiles={n:oracle_profile(act,g,legend) for n,g in graphs.items() if n not in ('O2','M2','W')}
                clocks=[oracle_clocks(d,coords,family,mode) for mode in (0,1,3)]
                checker=diag_checker(spec,d)
                got=attention(d,coords,family+'CLOCK',0);torch.cuda.synchronize();ok,msg=checker(got);assert ok,(B,L,'clock',msg);del checker
            graphs.clear();outs.clear();del d,coords,qkv;gc.collect();torch.cuda.empty_cache()
        RESULTS.append(dict(B=B,L=L,seeds=list(SEEDS),times_ns=times,TASK=gates,resources=resources,profiles=profiles,clocks=clocks))
    archive=dict(results=RESULTS,names=legend,source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in SOURCES.items()},elapsed_s=time.monotonic()-started,
        scope='Altered-work oracles only; C precomputes input-derived addresses outside timing and is never a candidate. Same loaded M kernel for all flags. PDL inputs ready; producer-context wait remains separate.',
        clock_phases=['entry','first_copies_issued','PDL_begin','PDL_end','Q_ready','history_done','new_tile_done','partials_ready','merge_gate_done'],
        clock_cycles=['page_load_issue','KV_copy_issue_including_address_dependency','copy_wait','V_transpose','QK','softmax_pack','PV'])
    blob=base64.b85encode(lzma.compress(json.dumps(archive,separators=(',',':')).encode())).decode();assert len(blob)<8600,('oracle archive cap',len(blob))
    return ['META P5 four-load sliding oracle: native D8/D4 and RAW2 controls; same-binary coords/constant-softmax/no-combine/loads-only/zero-byte/empty/ready-wait flags; no oracle promotes.',
            'SLIDING_ORACLE_XZ85 '+blob,'END loads='+str(len(RESULTS))+' elapsed='+str(round(time.monotonic()-started,2))]
