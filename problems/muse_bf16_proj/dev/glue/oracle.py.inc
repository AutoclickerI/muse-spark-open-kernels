# Synthetic untimed math guards. Never invoked by runtime projection calls.
def gb_state(weights):
    dev=weights['o'].device
    gen=torch.Generator(device=dev).manual_seed(838291)
    def vector(name,scale):
        return weights[name] if name in weights else (torch.randn(4096 if name=='pre_expert_norm' else 8192,
            device=dev,generator=gen)*scale).bfloat16()
    def coefficients(gate):
        gate=gate.float()/0.3;b=torch.sigmoid(gate)
        return torch.sqrt(torch.clamp(torch.sigmoid(-gate)*(1+b),min=1.e-3)),b
    a1,b1=coefficients(vector('attn_residual_gate',.3));a2,b2=coefficients(vector('ffn_residual_gate',.3))
    hi=weights['router'].bfloat16();lo=(weights['router'].float()-hi.float()).bfloat16()
    state={k:weights[k].contiguous() for k in ('qkv_gate','o','unpooler')}
    state.update(pool_router=torch.cat((weights['pooler'],hi,lo),0).contiguous(),
        unpooler_slice=weights['unpooler'][:1024].contiguous(),alpha1=a1.contiguous(),beta1=b1.contiguous(),
        alpha2=a2.contiguous(),beta2=b2.contiguous(),pre_ffn_w=1+vector('pre_ffn_norm',.1).float(),
        pre_expert_w=1+vector('pre_expert_norm',.1).float(),post_ffn_w=1+vector('post_ffn_norm',.1).float(),
        next_pre_attn_w=1+vector('next_pre_attn_norm',.1).float(),router_bias=(weights['router_bias'].float().contiguous()
            if 'router_bias' in weights else torch.randn(256,device=dev,generator=gen)*.035))
    return state


def gb_reference(phase,x,stream,state,data):
    if phase=='pool':
        p=x.float()@state['pool_router'].float().T
        pool=p[:,:4096].bfloat16().float()
        pooled=(pool*torch.rsqrt(pool.square().mean(-1,keepdim=True)+1.e-5)*state['pre_expert_w']).bfloat16()
        scores=torch.sigmoid((p[:,4096:4352]+p[:,4352:])*0.1767766952966369)
        biased=scores+state['router_bias'];ids=biased.topk(8,dim=-1).indices.to(torch.int32)
        w=scores.gather(1,ids.long());w=w/(w.sum(-1,keepdim=True)+1.e-15)
        lid=(ids-data['local_start']+256)%256;lid=torch.where(lid<32,lid,-1)
        return (pooled,ids,w,lid),{'scores':scores,'biased':biased}
    b2=phase=='b2';name=('unpooler_slice' if x.shape[0]<=8 else 'unpooler') if b2 else 'o'
    f=(x.float()@state[name].float().T).bfloat16().float()
    if f.shape[1]<8192:f=torch.nn.functional.pad(f,(0,8192-f.shape[1]))
    if b2:f=f*state['post_ffn_w']
    a,b,pre=('alpha2','beta2','next_pre_attn_w') if b2 else ('alpha1','beta1','pre_ffn_w')
    st=state[a]*stream+state[b]*f*torch.rsqrt(f.square().mean(-1,keepdim=True)+1.e-8)
    hidden=(st*torch.rsqrt(st.square().mean(-1,keepdim=True)+1.e-5)*state[pre]).bfloat16()
    return (hidden,st),{}


def gb_rel(actual,expected,limit=.01):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype or not bool(torch.isfinite(actual).all()):
        raise ValueError('glue dtype/shape/nonfinite')
    rel=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1.e-12))
    if rel>limit:raise ValueError(f'glue rel={rel:.6f}>{limit}')
    return rel


def gb_check(phase,actual,expected,meta,stream,state,data):
    err=gb_rel(actual[0],expected[0])
    if phase!='pool':
        err=max(err,gb_rel(actual[1],expected[1]))
        # Residual carry must not hide errors in the branch contribution.
        alpha=state['alpha2' if phase=='b2' else 'alpha1']
        err=max(err,gb_rel(actual[1]-alpha*stream,expected[1]-alpha*stream))
        return err
    ids=actual[1].long();scores=meta['scores'];biased=meta['biased']
    if bool((ids<0).any()) or bool((ids>=256).any()):raise ValueError('route id range')
    sorted_ids=ids.sort(-1).values
    if bool((sorted_ids[:,1:]==sorted_ids[:,:-1]).any()):raise ValueError('duplicate route')
    best=biased.topk(8,dim=-1);ninth=biased.topk(9,dim=-1).values[:,-1:]
    if float((best.values[:,-1:]-biased.gather(1,ids)).max())>.05:raise ValueError('public routing gap')
    start=int(data['local_start'].item());local=(best.indices-start)%256<32
    kept=(best.indices[:,:,None]==ids[:,None,:]).any(-1)
    if bool((local&~kept&(best.values-ninth>.03)).any()):raise ValueError('public required local expert')
    own_local=(ids-start)%256<32
    if int(own_local.sum())<int(local.sum())*.92-4:raise ValueError('public local route count')
    own_distinct=ids[own_local].unique().numel();ref_distinct=best.indices[local].unique().numel()
    if own_distinct<ref_distinct*.85-2:raise ValueError('public distinct local experts')
    wanted=scores.gather(1,ids);wanted=wanted/(wanted.sum(-1,keepdim=True)+1.e-15)
    err=max(err,gb_rel(actual[2],wanted))
    lid=(ids-start+256)%256;lid=torch.where(lid<32,lid,-1).to(torch.int32)
    if not torch.equal(actual[3],lid):raise ValueError('local expert map')
    return err
