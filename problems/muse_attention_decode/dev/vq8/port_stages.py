"""Sliding-only two/three-stage attention gathers; keep four stages for full context.
Smaller rings let more split CTAs fit on each SM, so the cluster planner can shorten sliding splits.
"""
import partial_port

def cell(stages):
    assert stages in (2,3)
    cu=partial_port.source()
    cu=cu.replace('launch_t<1, 4>(p, s)',f'(p.window ? launch_t<1, {stages}>(p, s) : launch_t<1, 4>(p, s))')
    cu=cu.replace('int64_t occupancy(int64_t q_len)', 'int64_t occupancy(int64_t q_len, int64_t window)')
    cu=cu.replace('occupancy_t<1, 4>()', f'(window ? occupancy_t<1, {stages}>() : occupancy_t<1, 4>())')
    cu=cu.replace('int64_t max_cluster(int64_t q_len, int64_t cs)', 'int64_t max_cluster(int64_t q_len, int64_t cs, int64_t window)')
    cu=cu.replace('max_cluster_t<1, 4>((int)cs)', f'(window ? max_cluster_t<1, {stages}>((int)cs) : max_cluster_t<1, 4>((int)cs))')
    cpp=partial_port.literals()['ATTN_PORT_CPP_SRC'].replace('double scale_log2, int64_t stream)',
        'double scale_log2, int64_t proj_splits, int64_t split_stride, int64_t qkv_width, int64_t stream)')
    cpp=cpp.replace('max_cluster(int64_t q_len, int64_t cs)', 'max_cluster(int64_t q_len, int64_t cs, int64_t window)')
    cpp=cpp.replace('occupancy(int64_t q_len)', 'occupancy(int64_t q_len, int64_t window)')
    drv=partial_port.DRIVER.replace('ext.occupancy(L)', 'ext.occupancy(L,window)').replace('ext.max_cluster(L,c)', 'ext.max_cluster(L,c,window)')
    for name in ('_PP_MOD','partial_port_ext','attention_partial_port','PARTIAL_CUDA','PARTIAL_FAST_CUDA','PARTIAL_CPP'):
        drv=drv.replace(name,name+f'_s{stages}')
    return (f'PARTIAL_CUDA_s{stages} = '+repr(cu)+f'\nPARTIAL_FAST_CUDA_s{stages} = PARTIAL_CUDA_s{stages}\n'+
            f'PARTIAL_CPP_s{stages} = '+repr(cpp)+'\n'+drv)
