def layer_weights(seed=5353):
    """Synthetic weights from the published TASK distribution; no reference import."""
    dev='cuda';g=torch.Generator(device=dev).manual_seed(seed)
    def fp4(rows):
        codes=torch.randint(0,256,(32,rows,2048),device=dev,generator=g,dtype=torch.uint8)
        z=torch.randn((32,rows,256),device=dev,generator=g)
        sf=torch.exp2(7.+.4*z).clamp(8.,448.).to(torch.float8_e4m3fn)
        msq=4.**7*math.exp(2*(math.log(2)*.4)**2)
        gs=torch.full((32,),1./math.sqrt(8.6*msq*4096),device=dev)
        return codes,sf,gs
    w13,s13,g13=fp4(8192);w2,s2,g2=fp4(4096)
    g13*=torch.exp(torch.randn(32,device=dev,generator=g));g2*=torch.exp(torch.randn(32,device=dev,generator=g))
    def dense(n,k):return (torch.randn((n,k),device=dev,generator=g)/math.sqrt(k)).to(torch.bfloat16)
    def vec(n,std=.1):return (std*torch.randn(n,device=dev,generator=g)).to(torch.bfloat16)
    o=dense(8192,1024);attn=vec(8192,.3);pre=vec(8192)
    # The public stream basis is generated on CPU from the fixed seed5354.
    cpu=torch.Generator().manual_seed(5354);mean=torch.randn(8192,generator=cpu).to(dev)
    router=.045*torch.randn((256,8192),device=dev,generator=g)+.07*torch.randn((256,1),device=dev,generator=g)*mean/math.sqrt(8192)
    bias=.055*torch.randn(256,device=dev,generator=g)
    return dict(qkv_gate=dense(2304,8192),o=o,attn_residual_gate=attn,pre_ffn_norm=pre,router=router,router_bias=bias,
        pooler=dense(4096,8192),pre_expert_norm=vec(4096),w13=w13,w13_sf=s13,w13_gs=g13,w2=w2,w2_sf=s2,w2_gs=g2,
        post_expert_norm=vec(4096),unpooler=dense(8192,4096),post_ffn_norm=vec(8192),ffn_residual_gate=vec(8192,.3),next_pre_attn_norm=vec(8192))
