_br_project_base=_best7_project
def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    if not (plan.get('br_pub') and name=='unpooler'):
        return _br_project_base(x,name,state,plan,out_dtype)
    t,k=x.shape;n=state[name].shape[0]
    assert t in (16,32) and (n,k)==(8192,4096)
    assert _GEMV_CHOICES[(t,n,k)]==('bal',64,2,4),'publication must preserve the S14 producer recipe'
    p=torch.empty((2,t,n),device=x.device,dtype=torch.float32);flags=torch.empty((2,n//64),device=x.device,dtype=torch.int32)
    _launch(_br_publish_kernel,(n//64,2),x,state[name],p,flags,t,x.stride(0),n,k,2,BN=64,BK=128,TB=max(16,t),NS=4,num_warps=4)
    return p,2,flags
