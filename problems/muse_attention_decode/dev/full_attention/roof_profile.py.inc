def profile(ns,data,modules,check):
    cfg=data['config'];B,L=cfg['batch'],cfg['q_len'];dev=data['qkv'].device
    sms=torch.cuda.get_device_properties(dev).multi_processor_count
    stream=torch.cuda.current_stream().cuda_stream
    if L==1:
        if B>1 and 'q1_arms' in modules:
            other={k:v for k,v in modules.items() if k!='q1_arms'}
            return {name:profile(arm,data,{**other,'q1_clock':mod},lambda d,y,n,name=name:check(d,y,n+'_'+name))
                    for name,(arm,mod) in modules['q1_arms'].items()}
        # d1's actual partial-input CUDA, regular attn_main at d16/32.
        mod=modules['partial_clock' if B==1 else 'q1_clock']
        original=ns['partial_port_ext']() if B==1 else ns['_attn_port_ext']()
        occ=(max(1,original.occupancy(1)),{cs:original.max_cluster(1,cs) for cs in range(1,17)})
        splits,cluster=ns['_attn_main_plan'](B,1,data['page_table'].shape[1]*cfg['page_size'],1,0,sms,occ)
        rows=B*(splits//cluster)*16
        po=torch.empty(rows*64,device=dev);ml=torch.zeros(rows*2+B*splits*4*16*2,device=dev)
        count=torch.zeros(B*cluster,device=dev,dtype=torch.int32);out=torch.empty((B,L,1024),device=dev,dtype=torch.bfloat16)
        common=[data[k].data_ptr() for k in ('qkv','positions','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')]
        tail=[0,po.data_ptr(),ml.data_ptr(),count.data_ptr(),po.data_ptr(),1,out.data_ptr(),B,L,data['qkv'].stride(-2),16,2,data['page_table'].shape[1],cfg['page_size'],0,splits,cluster,cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634]
        if B==1:tail +=[0,0,data['qkv'].shape[-1]]
        mod.attn_forward(*common,*tail,stream)
        check(data,out,'profile_q1')
        c=ml[rows*2:].view(torch.int64).view(-1,16).cpu();live=c[c[:,10]>0];nt=live[:,10].double()
        med=lambda x:float(x.double().median())
        return dict(splits=splits,cluster=cluster,tiles=[int(nt.min()),int(nt.max())],
            cycles_per_tile={name:med(live[:,i]/nt) for i,name in ((3,'copy_wait'),(4,'CTA_sync'),(5,'page_issue'),(6,'QK'),(7,'softmax'),(8,'PV'))},
            prologue_cycles=med(live[:,2]),epilogue_cycles=med(live[:,9]),loop_cycles=med(live[:,11]))
    # VQ8's built-in clocks preserve the actual N128 stage4 family.
    result={}
    for name,mod in modules['vq_clocks'].items():
        if B==1 and not name.startswith('H'):continue # Only H arms replace FA3 at v1.
        one=name.startswith('H');grid=sms;segments=B
        ng=grid*(2 if name=='H2' else 1);nseg=segments*(2 if one else 1)
        # H1's actual prefix is smaller; H2 equals S15's allocation.
        prefix=(ng+nseg)*(64 if one else 128)*68+256
        nrows=ng+16*L*segments
        p=torch.zeros(prefix+nrows*64,device=dev);out=torch.empty((B,L,1024),device=dev,dtype=torch.bfloat16)
        mod.vq8_forward(*[data[k].data_ptr() for k in ('qkv','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')],
            0,p.data_ptr(),out.data_ptr(),B,L,16,2,data['qkv'].stride(-2),data['page_table'].shape[1],cfg['page_size'],0,0,2 if B==1 else 8,grid,4,
            cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,0,1,stream)
        check(data,out,'profile_'+name)
        c=p[prefix:].view(torch.int64).view(-1,32).cpu();main=c[:ng];main=main[(main[:,10]>0)&(main[:,10]<(1<<32))];n=main[:,10].double()
        med=lambda x:float(x.double().median());comb=c[ng:];comb=comb[comb[:,2]>0]
        result[name]=dict(resources=mod.vq8_resources(4),ctas=len(main),tiles=[int(n.min()),int(n.max())],
            cycles_per_tile={key:med(main[:,i]/n) for i,key in ((19,'QK_wait'),(20,'softmax'),(21,'PV_wait'),(22,'MMA_issue'),(23,'V_publish'),(24,'KV_page_issue'))},
            combine_wait_us=med(comb[:,1]-comb[:,0])/1000,combine_work_us=med(comb[:,2]-comb[:,1])/1000,
            main_span_us=float((main[:,3].max()-main[:,0].min()).double())/1000)
    return result
