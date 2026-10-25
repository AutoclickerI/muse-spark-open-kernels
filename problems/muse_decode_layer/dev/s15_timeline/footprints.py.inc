def t15_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,(tuple,list)):return sum(t15_bytes(v) for v in value)
    return 0


def t15_work(data,state,io,ns):
    cfg=data['config'];t=cfg['tokens'];q=cfg['q_len'];work={};details={}
    def add(role,reads=0,writes=0,bf=0,fp=0,**info):
        work[role]=dict(read_bytes=int(reads),write_bytes=int(writes),bf16_flops=int(bf),fp8_flops=int(fp));details[role]=info
    for role,(args,out) in io.items():
        if role in ('qkv_gate','qkv3','o','pool_router','unpooler','unpooler_slice'):
            x=args[0];w=state[role];n,k=w.shape;wb=t15_bytes(w);encoding='BF16'
            if 'nf_port' in state and (t,role) in ns['NF_PORT_SELECTED']:
                p=state['nf_port']['plans'].get((t,role))
                if p and 'wp' in p:wb=t15_bytes(p['wp'])+t15_bytes(p['esc']);encoding='NF lossless codes+metadata'
                elif p and 'w' in p:wb=t15_bytes(p['w']);encoding='NF original-bit layout'
            elif (t,role) in ns['CD_ROWS'] and 'coded_dense' in state:
                p=state['coded_dense'][role];wb=sum(t15_bytes(p[k]) for k in ('coded','scales','eoff','esc'));encoding='lossless CD codes+metadata'
            add(role,wb+t15_bytes(x),t15_bytes(out[0]),2*t*n*k,weights_bytes=wb,encoding=encoding,splits=out[1],shape=[t,n,k])
        elif role=='gate':
            weight=state['gate_rows'];add(role,t15_bytes(weight)+t15_bytes(args[0])+t15_bytes(args[1]),t15_bytes(out),2*t*8192*1024,encoding='BF16 gate rows')
        elif role in ('b1','b2'):
            add(role,t15_bytes(args[0][0])+t15_bytes(args[1])+8192*4*(4 if role=='b2' else 3),t15_bytes(out),0,0,model='partial+residual+weights read once; FP32 reduction throughput unmodeled')
        elif role=='route':
            add(role,t15_bytes(args[0][0])+t15_bytes(state['router_bias'])+t15_bytes(state['pre_expert_w'])+4,t15_bytes(out),0,0,model='pool/router partials+weights read once; selection throughput unmodeled')
        elif role=='attn':
            seq=data['seq_lens'].cpu().tolist();window=cfg['window'];hq=cfg['num_q_heads'];hkv=cfg['num_kv_heads'];dim=cfg['head_dim']
            unique_keys=sum(min(s+q,window+q-1) if window else s+q for s in seq)
            causal_keys=sum(min(s+j+1,window) if window else s+j+1 for s in seq for j in range(q))
            kv=unique_keys*hkv*dim*2;reads=kv+unique_keys*4+t15_bytes(args[0][0])+t15_bytes(data['seq_lens'])+8;flops=4*hq*dim*causal_keys
            name='attention_sliding' if window else 'attention_full'
            add(name,reads,t15_bytes(out),flops if q==1 else 0,flops if q!=1 else 0,kv_bytes=kv,unique_keys=unique_keys,causal_query_keys=causal_keys,precision_model='FP16/BF16 decode MMA' if q==1 else 'FP8 QK/PV',model='unique cached K/V and page entries; excludes split rereads/scratch; norm/softmax scalar throughput unmodeled')
        elif role=='moe':
            lids=args[3].cpu().flatten().tolist();counts=[lids.count(i) for i in range(32)];details['expert_counts']=counts
            buckets={'moe_small':[c for c in counts if c>0 and (t<=32 or c<=8)],'moe_light':[c for c in counts if t>32 and 8<c<=16],'moe_heavy':[c for c in counts if t>32 and c>16]}
            w13=8192*4096*9//16;w2=4096*4096*9//16
            for name,cells in buckets.items():
                pairs=sum(cells);experts=len(cells)
                # NVFP4 codes plus E4M3 scales, one unique expert read. Intermediate
                # activations remain BF16; final slot y is FP32. This is a lower
                # bound on traffic, not the tiled engine's reread count.
                reads=experts*(w13+w2+8)+pairs*(4096*2+4096*2)+experts*4096*2
                writes=pairs*(4096*2+4096*4+64*4)
                add(name,reads,writes,2*pairs*4096*(8192+4096),experts=experts,pairs=pairs,model='unique NVFP4 weights; compulsory slot activations, not HBM counters')
            if t<=32:
                # The fused cell includes its internal setup/norm/combine work.
                work['moe_small']['read_bytes']+=t15_bytes(args[2])+t15_bytes(args[3]);work['moe_small']['write_bytes']+=t15_bytes(out)
            else:
                add('moe_meta',t*8*4,t*8*4+32*4*10,model='routing indices/counters lower bound')
                add('moe_combine',sum(counts)*(4096*4+64*4)+t15_bytes(args[2])+t15_bytes(args[3]),t15_bytes(out),model='slot y, partial norms, route weights and final BF16 output')
    return work,details
