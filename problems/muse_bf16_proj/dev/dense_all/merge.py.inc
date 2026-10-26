# Fixed-cost arm: QKV and gate share input X and one projection launch.
# The original gate-split arm remains the paired control; only shape decides.
_da_original_split_on=_gate_split_on
DA_MERGE_POINTS={(1,True),(1,False),(16,True),(32,False)}
def _gate_split_on(data,state,plan):
    cfg=data['config']
    if (cfg['tokens'],bool(cfg['window'])) in DA_MERGE_POINTS:return False
    return _da_original_split_on(data,state,plan)
