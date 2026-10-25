"""Cluster-route MoERouteMeta producer ABI for P3a's consumer experiment.

No promotion: small-M must consume data['route_meta'] and measure the pair.
Enable state['cluster_route_meta_on'] in the paired lab only.
"""
import hashlib
from pathlib import Path
import sys

BLOCK = '''

def cluster_route_meta_prepare(state, device):
    state['cluster_route_meta'] = {
        m: (torch.empty(736, device=device, dtype=torch.int32),
            torch.zeros(1, device=device, dtype=torch.int32)) for m in (1,8,16,32)}


_cluster_route_meta_base_route = _best7_route


def _best7_route(pr, state, data, plan, sync=None):
    p,s = pr
    t = p.shape[1]
    if not state.get('cluster_route_meta_on',False) or t not in state['cluster_route_meta']:
        return _cluster_route_meta_base_route(pr,state,data,plan,sync)
    meta,arrive = state['cluster_route_meta'][t]
    dev = p.device
    x = torch.empty((t,4096),device=dev,dtype=torch.bfloat16)
    ids = torch.empty((t,8),device=dev,dtype=torch.int32)
    wts = torch.empty((t,8),device=dev,dtype=torch.float32)
    lids = torch.empty_like(ids)
    _glue_cl_ext().route_cl(p.data_ptr(),int(p.dtype==torch.float32),s,p.stride(0),p.stride(1),
        state['router_bias'].data_ptr(),state['pre_expert_w'].data_ptr(),data['local_start'].data_ptr(),
        x.data_ptr(),ids.data_ptr(),wts.data_ptr(),lids.data_ptr(),meta.data_ptr(),arrive.data_ptr(),
        t,8,256,torch.cuda.current_stream(dev).cuda_stream)
    return x,ids,wts,lids
'''


def patch(src):
    anchor = '    _gate_split_prepare(state,device)\n'
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + '    cluster_route_meta_prepare(state,device)\n')
    old = "    data=dict(hidden_states=x,topk_ids=lids,topk_weights=wts,pool_partials=pr[0],pool_splits=pr[1],\n"
    new = old + "        route_meta=state['cluster_route_meta'][x.shape[0]][0] if state.get('cluster_route_meta_on',False) and x.shape[0]<=32 else None,\n"
    assert src.count(old) == 1
    return src.replace(old, new) + BLOCK


if __name__ == '__main__':
    out = patch(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2],hashlib.sha256(out.encode()).hexdigest())
