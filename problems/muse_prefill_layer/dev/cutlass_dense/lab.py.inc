
def pfc_lt_module():
    from torch.utils.cpp_extension import load_inline
    previous=os.environ.get('TORCH_CUDA_ARCH_LIST')
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
    try:
        return load_inline(name='prefill_exact_lt32_'+hashlib.sha256(PFC_LT_SOURCE.encode()).hexdigest()[:12],
            cpp_sources='',cuda_sources=PFC_LT_SOURCE,no_implicit_headers=True,
            extra_cuda_cflags=['-O3','-std=c++20'],extra_ldflags=['-lcublasLt'],verbose=False)
    finally:
        if previous is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
        else:os.environ['TORCH_CUDA_ARCH_LIST']=previous


def lab(deadline,weights,state=None):
    begin=time.monotonic()
    # Wrapper deadline is from BEFORE host prepare: never reset that clock.
    deadline=min(deadline,begin+270)
    rows=[];rejects=[]
    try:module=vgc_module();lt=pfc_lt_module()
    except Exception as exc:return [f'PFC_REJECT compile {type(exc).__name__}:{str(exc)[-350:]}']
    compile_s=time.monotonic()-begin
    device=weights['qkv_gate'].device
    hi=weights['router'].bfloat16();lo=(weights['router']-hi.float()).bfloat16()
    st={'qkv':weights['qkv_gate'],'o':weights['o'],'pooler':weights['pooler'],
        'unpooler':weights['unpooler'],'router_cat':torch.cat((hi,lo),0).contiguous()}
    st['dense_transposed']={key:st[key].T.contiguous() for key in PFC_SHAPES}
    st['sms']=torch.cuda.get_device_properties(device).multi_processor_count
    st['grouped_mod']=lt;st['lt_algorithms']={}
    st['lt_workspace']=torch.empty(256<<20,device=device,dtype=torch.uint8)
    flush=torch.zeros((1<<30)//4,device=device,dtype=torch.float32)
    rows.append(f'PFC_META compile_s={compile_s:.3f} sms={st["sms"]} incumbent=0d738799802340fd; exact Lt32/256MiB; cold1GiB, fresh3, paired20; TFLOPS=2MNK/us/1e6; peak=989')
    # Allocate time to every shape, with the largest absolute lever first.
    keys=[(m,key) for m in (16384,753,3012) for key in PFC_SHAPES]
    shares={(m,key):(2 if key in ('pooler','unpooler') else 1) for m,key in keys}
    left=sum(shares.values())
    for m,key in keys:
        now=time.monotonic();stop=now+max(0,deadline-now-3)*shares[m,key]/left
        left-=shares[m,key]
        if now>=deadline-3:rows.append(f'PFC_BUDGET {m} {key}');continue
        old=None;cu=None
        try:
            w=st[key];n,k=w.shape;router=key=='router_cat';dtype=torch.float32 if router else torch.bfloat16
            banks=pfc_banks(m,k,device);x=banks[0].clone()
            refs=[pfc_reference(fresh,w) for fresh in banks]
            router_refs=[pfc_reference(fresh,weights['router']) for fresh in banks] if router else None
            cu=_PgGraph(lambda:_score_dense_run(x,st,key,('cublas',)))
            choice=PFC_INCUMBENTS[key,m]
            if choice[0]=='lt':st['lt_algorithms'][key,m]=lt.lt_options(w,x,cu.out)
            old=_PgGraph(lambda:_score_dense_run(x,st,key,choice))
            for fresh,ref in zip(banks,refs):
                x.copy_(fresh);cu.graph.replay();old.graph.replay()
                pfc_check(cu.out,ref,router);pfc_check(old.out,ref,router)
            cu_us,inc_us,_=pfc_pair(cu,old,x,banks,flush,20,min(stop,deadline-2))
            attempts=0;bad=0;trials=[];tile_best={};mode_best={}
            coarse_stop=time.monotonic()+.72*max(0,stop-time.monotonic())
            for recipe in pfc_choices(m,n,k,st['sms']):
                if time.monotonic()>=coarse_stop:break
                plan=None;attempts+=1
                try:
                    plan=module.plan(m,n,k,dtype,recipe,device)
                    def run():
                        out=torch.empty((m,n),device=device,dtype=dtype)
                        return plan.run(x,w,out)
                    with _PgGraph(run) as graph:
                        error=0.;worst=0.
                        for i,(fresh,ref) in enumerate(zip(banks,refs)):
                            x.copy_(fresh);graph.graph.replay()
                            rel,row=pfc_check(graph.out,ref,router);error=max(error,rel);worst=max(worst,row)
                            if router:
                                pfc_check(graph.out[:,:256]+graph.out[:,256:],router_refs[i],True)
                        t0,t1,ratio=pfc_pair(old,graph,x,banks,flush,4,min(stop,deadline-2))
                    trials.append((ratio,recipe,error,worst,plan.workspace.numel(),plan.info))
                    for mapping,family in ((tile_best,recipe[0]),(mode_best,recipe[1])):
                        if family not in mapping or t1<mapping[family]:mapping[family]=t1
                except Exception as exc:
                    bad+=1
                    if len(rejects)<4:rejects.append(f'PFC_REJECT {m} {key} {recipe} {type(exc).__name__}:{str(exc).splitlines()[0][:100]}')
                finally:
                    if plan is not None:plan.close()
            finals=sorted(trials)[:2]
            for mode in (1,2,3):
                family=[row for row in trials if row[1][1]==mode]
                if family and min(family) not in finals:finals.append(min(family))
            best=None;final_rows=[]
            for _,recipe,error,worst,size,info in finals:
                if time.monotonic()>=min(stop,deadline-2):break
                plan=None
                try:
                    plan=module.plan(m,n,k,dtype,recipe,device)
                    def run_final():
                        out=torch.empty((m,n),device=device,dtype=dtype)
                        return plan.run(x,w,out)
                    with _PgGraph(run_final) as graph:
                        t0,t1,ratio=pfc_pair(old,graph,x,banks,flush,20,min(stop,deadline-2))
                        for i,(fresh,ref) in enumerate(zip(banks,refs)):
                            x.copy_(fresh);graph.graph.replay();pfc_check(graph.out,ref,router)
                            if router:pfc_check(graph.out[:,:256]+graph.out[:,256:],router_refs[i],True)
                    final_rows.append([list(recipe),round(t0,3),round(t1,3),round(ratio,5)])
                    if best is None or ratio<best[0]:best=(ratio,recipe,t0,t1,error,worst,size,info)
                except Exception as exc:
                    bad+=1
                    if len(rejects)<4:rejects.append(f'PFC_REJECT final {m} {key} {type(exc).__name__}:{str(exc)[:100]}')
                finally:
                    if plan is not None:plan.close()
            result={'m':m,'key':key,'cu':round(cu_us,3),'inc':round(inc_us,3),'c':choice,
                    'pick':list(best[1]) if best and best[0]<1 else None,
                    'best':([round(best[2],3),round(best[3],3),round(best[0],5),
                             round(2*m*n*k/best[3]/1e6,2),float(f'{best[4]:.3g}'),float(f'{best[5]:.3g}'),
                             best[6],list(best[7])] if best else None),
                    'tile_us':[round(tile_best[t],2) if t in tile_best else None for t in range(8)],
                    'mode_us':[round(mode_best[t],2) if t in mode_best else None for t in range(4)],
                    'finals':final_rows,'tried':attempts,'rejected':bad}
            rows.append('PFC_ROW '+json.dumps(result,separators=(',',':')))
        except Exception as exc:
            rows.append(f'PFC_REJECT shape {m} {key} {type(exc).__name__}:{str(exc)[:140]}')
        finally:
            if old is not None:old.close()
            if cu is not None:cu.close()
    # Preserve the 15 measurement rows within the wrapper's 9000-char output cap.
    rows.extend(rejects)
    rows.append(f'PFC_END elapsed_s={time.monotonic()-begin:.3f} left_s={deadline-time.monotonic():.3f}')
    return rows
