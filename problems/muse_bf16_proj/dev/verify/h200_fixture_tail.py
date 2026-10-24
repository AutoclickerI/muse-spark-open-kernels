# Actual H200-shortlist integration; force all receipt recipes after exercising selection.
_vp_h200_prepare=prepare
_vp_h200_project=project
project=verify_proj_bind(_vp_h200_project)

def prepare(weights):
    started=time.monotonic();state=_vp_h200_prepare(weights)
    rows=verify_proj_prepare(state,_vp_h200_project,started+240)
    bank=state['_verify_proj'];bank['lt']=_pg_native()
    if 'workspace' not in bank:bank['workspace']=torch.empty(4<<30,device=state['o'].device,dtype=torch.uint8)
    checked=0
    for (m,name),choices in VERIFY_PROJ_RECIPES.items():
        recipe=choices[0];_,nn,ordinal=recipe;w=state[name]
        dtype=torch.float32 if name=='pool_router' else torch.bfloat16
        codes,_=bank['lt'].options(m,w.shape[0],w.shape[1],nn,dtype==torch.float32,bank['workspace'].numel())
        assert ordinal<len(codes),(m,name,ordinal,len(codes))
        bank['layouts'][(m,name,nn)]=(w.T.contiguous() if nn else w,codes)
        bank['plan'][(m,name)]=(recipe,w.data_ptr(),tuple(w.shape),tuple(w.stride()),dtype)
        x=torch.randn(m,w.shape[1],device=w.device,dtype=torch.bfloat16)
        with _PgGraph(lambda:_vp_materialize(verify_project(x,w,name,state,out_dtype=dtype))) as graph:
            for _ in range(3):
                x.normal_();graph.graph.replay();vp_check(graph.out,x.float()@w.float().T);checked+=1
    for row in rows:print(row,flush=True)
    print('VP_H200_RECIPE_FORCED keys='+str(len(bank['plan']))+' fresh='+str(checked),flush=True)
    return state
