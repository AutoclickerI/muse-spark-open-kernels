DC_SCREEN_ARM='sparse'

def prepare(weights):
    return dc_prepare_roles(weights)


def kernel_fn(data,state):
    x=data['x'];name=data['config']['layer'];t=x.shape[0];sparse=DC_SCREEN_ARM=='sparse'
    if name=='router':
        # Original precision split; each half has its own independent lossless bank.
        if t not in (1,8,16,32,128,256):
            parts=[]
            for role in ('router_hi','router_lo'):
                w=state['roles'][role]
                if sparse:w=dc_unpack(state['sparse'][role],torch.empty_like(w))
                parts.append(torch.mm(x,w.T,out_dtype=torch.float32))
            return parts[0]+parts[1]
        a=_materialize(dc_project(x,'router_hi',state,sparse));b=_materialize(dc_project(x,'router_lo',state,sparse))
        return a.float()+b.float()
    role='unpooler_slice' if name=='unpooler' and t<=8 else name
    if role=='pooler':
        # Board requests pooler alone; layer's fused N4608 arm is in the payload.
        role='pooler'
    if t not in (1,8,16,32,128,256):
        # Nearby shapes use the same timed exact decode + full dense product.
        if sparse:
            c=state['sparse'][role];w=torch.empty_like(state['roles'][role]);dc_unpack(c,w)
        else:w=state['roles'][role]
        return torch.mm(x,w.T).to(torch.bfloat16)
    return _materialize(dc_project(x,role,state,sparse)).to(torch.bfloat16)
