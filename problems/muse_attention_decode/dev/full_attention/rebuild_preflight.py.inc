    # Every structural arm also sees all official whole-layer TASK fixtures,
    # eagerly and after two captured replays; these are correctness only.
    official=[dict(batch=1,ctx=10,layer='sliding'),dict(batch=1,ctx=3000,layer='full'),
        dict(batch=8,ctx=2047,layer='sliding'),dict(batch=9,ctx=5000,layer='sliding',spread='mixed',page=1),
        dict(batch=3,ctx=6000,layer='full',q_len=8,dist='outliers'),dict(batch=5,ctx=5000,layer='sliding',q_len=8,spread='mixed',page=1),
        dict(batch=33,ctx=20000,layer='full',spread='mixed',page=1),dict(batch=2,ctx=300000,layer='sliding',q_len=8),
        dict(batch=1,ctx=1048576,layer='full',q_len=8),dict(batch=4,ctx=7000,layer='sliding',q_len=8,page=1,kv_scale=.015625),
        dict(batch=1,ctx=80000,layer='full',dist='soft',page=1),dict(batch=4,ctx=6000,layer='sliding',q_len=8,dist='soft'),
        dict(batch=16,ctx=20000,layer='full',spread='mixed',dist='soft',page=1)]
    preflight_checks=0;preflight_fail=[]
    for i,spec in enumerate(official):
        data=generator['generate_input'](**spec,seed=711991+i)
        ids=base['kernel_fn'](data,state)[2]
        data['local_start'].copy_((32*torch.bincount((ids.long()//32).flatten(),minlength=8).argmax()).reshape(1).int())
        ff,bf=checker['ref_front'],checker['ref_back'];fc={};bc={}
        def front(d,w,query_dtype=torch.bfloat16):
            if query_dtype not in fc:fc[query_dtype]=ff(d,w,query_dtype)
            return fc[query_dtype]
        def back(f,w,ids):
            key=id(f),tuple(ids.reshape(-1).tolist())
            if key not in bc:bc[key]=bf(f,w,ids)
            return bc[key]
        checker['ref_front']=front;checker['ref_back']=back
        try:
            for name,ns in spaces.items():
                fn=lambda:ns['kernel_fn'](data,states[name])
                y=fn();torch.cuda.synchronize()
                ok,msg=checker['check_implementation'](data,checker['collect'](data,y),weights);preflight_checks+=1
                if not ok:preflight_fail.append([i,name,'eager',msg])
                g,y=graph(fn);g.replay();g.replay();torch.cuda.synchronize()
                ok,msg=checker['check_implementation'](data,checker['collect'](data,y),weights);preflight_checks+=1
                if not ok:preflight_fail.append([i,name,'replay',msg])
                del g,y
        finally:checker['ref_front']=ff;checker['ref_back']=bf
        data=ids=fc=bc=None;gc.collect();torch.cuda.empty_cache()
    assert preflight_checks==182 and not preflight_fail,('official TASK preflight',preflight_checks,preflight_fail)
