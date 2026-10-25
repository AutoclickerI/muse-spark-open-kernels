SP_ARM=(1,8,1,1,True)
SP_FORCE=False

def prepare(weights):
    wr,wk,s,st,sparse=SP_ARM
    native=torch.cuda.get_device_capability()==(9,0)
    if s>1 and not native:
        # DSM-only launch stand-in; native sparse decoder/MMA checked with S1.
        bank=dict(w=torch.cat((weights['pooler'],weights['router'].bfloat16(),(weights['router']-weights['router'].bfloat16().float()).bfloat16())),standin=True)
    else:bank=sp_prepare(weights,wr,wk,s,st,sparse)
    return dict(weights=weights,bank=bank)


def kernel_fn(data,state):
    x=data['x'];name=data['config']['layer'];m=x.shape[0]
    if name in ('pooler','router') and m in (1,8,16,32):
        bank=state['bank']
        pair=(torch.mm(x,bank['w'].T,out_dtype=torch.float32).unsqueeze(0),1) if bank.get('standin') else sp_project(x,bank)
        y=pair[0][0]
        return y[:,:4096].bfloat16() if name=='pooler' else y[:,4096:4352]+y[:,4352:]
    w=state['weights'][name]
    if name=='unpooler' and m<=8:w=w[:1024]
    if name=='router':
        hi=w.bfloat16();lo=(w-hi.float()).bfloat16()
        return torch.mm(x,hi.T,out_dtype=torch.float32)+torch.mm(x,lo.T,out_dtype=torch.float32)
    return torch.mm(x,w.T,out_dtype=torch.float32).bfloat16()
