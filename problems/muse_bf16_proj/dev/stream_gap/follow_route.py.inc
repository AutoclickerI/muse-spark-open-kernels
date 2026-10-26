# Routing checker copied verbatim from the published TASK reference; diagnostics only.
N_EXPERTS = 256

TOP_K = 8

LOCAL_EXPERTS = 32

ROUTE_TOL = 5e-2

LOCAL_SLACK = (0.08, 4)

DISTINCT_SLACK = (0.15, 2)

KEEP_TOL = (0.03, 0.04)

KEEP_TOKENS = 256

def _local_load(ids, local_start=0):
    """Routes to this GPU's experts (local_start .. local_start + 31), and how many distinct
    ones they touch."""
    local = ids[(ids - local_start) % N_EXPERTS < LOCAL_EXPERTS]
    return int(local.numel()), int(torch.unique(local).numel())

def sg_task_check_routing(ids, biased, local_start=0):
    """None if ids [T, 8] are an acceptable top-8 under the reference's biased scores."""
    if ids.min() < 0 or ids.max() >= N_EXPERTS:
        return "expert id out of range"
    ranked = ids.long().sort(-1).values
    if (ranked[:, 1:] == ranked[:, :-1]).any():
        return "a token routed to the same expert twice"
    best = biased.topk(TOP_K, dim=-1)
    gap = best.values[:, -1:] - biased.gather(1, ids.long())
    if gap.max() > ROUTE_TOL:
        t = int(gap.max(-1).values.argmax())
        return (f"token {t} routed to an expert whose biased score is {float(gap[t].max()):.2e} below its 8th best"
                f" (> {ROUTE_TOL:g})")
    ninth = biased.topk(TOP_K + 1, dim=-1).values[:, -1:]
    keep_tol = KEEP_TOL[0] if len(ids) <= KEEP_TOKENS else KEEP_TOL[1]
    kept = (best.indices[:, :, None] == ids.long()[:, None, :]).any(-1)
    local = (best.indices - local_start) % N_EXPERTS < LOCAL_EXPERTS
    lost = local & ~kept & (best.values - ninth > keep_tol)
    if lost.any():
        t, slot = (int(i) for i in lost.nonzero()[0])
        return (f"token {t} dropped expert {int(best.indices[t, slot])}, on this GPU, whose biased score is"
                f" {float(best.values[t, slot] - ninth[t]):.2e} above its 9th best (> {keep_tol:g})")
    routes, distinct = _local_load(ids.long(), local_start)
    ref_routes, ref_distinct = _local_load(best.indices, local_start)
    if routes < ref_routes * (1 - LOCAL_SLACK[0]) - LOCAL_SLACK[1]:
        return f"{routes} routes to this GPU's experts where the reference has {ref_routes}"
    if distinct < ref_distinct * (1 - DISTINCT_SLACK[0]) - DISTINCT_SLACK[1]:
        return f"{distinct} of this GPU's experts touched where the reference touches {ref_distinct}"
    return None

def sg_follow_route(out,x,router,bias,prew,pool_reference):
    T=torch;ids=out[1].long();ranked=ids.sort(-1).values
    assert int(ids.min())>=0 and int(ids.max())<256 and not bool((ranked[:,1:]==ranked[:,:-1]).any()),'unique expert ids'
    scores=T.sigmoid((x.double()@router.double().T).float()*.1767766952966369);biased=scores+bias
    error=sg_task_check_routing(ids,biased,0);assert error is None,error
    picked=scores.gather(1,ids);expected=picked/(picked.sum(-1,keepdim=True)+1e-15)
    assert float((out[2]-expected).norm()/expected.norm())<.03,'routing weights'
    assert T.equal(out[3],T.where(ids<32,ids,-1).int()),'local ids'
    p=pool_reference.float();norm=(p*T.rsqrt(p.square().mean(-1,keepdim=True)+1e-5)*prew).bfloat16();e=float((out[0].float()-norm.float()).norm()/norm.float().norm());assert e<.03,('pool norm',e)
    return e
