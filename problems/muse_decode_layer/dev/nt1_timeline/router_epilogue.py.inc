    if EMIT:
        tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0,$1;", "=r,r",
            [tl.arange(0,256)],dtype=tl.int32,is_pure=False,pack=1)
        tl.debug_barrier()
        if part==0:
            previous=tl.atomic_add(ARRIVE,1,sem='acq_rel')
            if previous==M-1:
                z=tl.arange(0,1024)
                tl.store(META+z,0,z<736)
                tl.debug_barrier()
                i=tl.arange(0,256)
                local_ids=tl.load(LIDS+i,i<M*8,-1)
                valid=(i<M*8)&(local_ids>=0)&(local_ids<32)
                counts=tl.histogram(local_ids,32,mask=valid)
                meta_e=tl.arange(0,32)
                ends=tl.cumsum(counts,0);blocks=(counts+7)//8;units=tl.cumsum(blocks,0)
                tl.store(META+4+meta_e,counts);tl.store(META+36+meta_e,ends-counts)
                tl.store(META+69+meta_e,blocks);tl.store(META+101+meta_e+1,units*512);tl.store(META+134+meta_e+1,units*256)
                tl.store(META+36+32,tl.sum(counts,0))
                tl.store(META,1);tl.store(META+1,M);tl.store(META+2,start);tl.store(META+3,tl.sum(counts,0))
                token_counts=tl.sum(tl.reshape(valid.to(tl.int32),(32,8)),1)
                tl.store(META+167+meta_e,token_counts)
                meta_keys=tl.where(valid,local_ids*256+i,2147483647)
                meta_ordered=tl.sort(meta_keys,descending=False);good=meta_ordered<32*256;slot=meta_ordered&255
                tl.store(META+199+i,tl.where(good,slot,-1))
                tl.store(META+455+i,-1)
                tl.debug_barrier()
                tl.store(META+455+slot,i,good)
                tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0,$1;", "=r,r",
                    [tl.arange(0,256)],dtype=tl.int32,is_pure=False,pack=1)
                tl.debug_barrier()
                tl.atomic_xchg(ARRIVE,0,sem='release')
