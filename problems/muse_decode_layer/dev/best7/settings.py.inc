# One row per scored case. Columns: glue, attention, MoE, projections.
# glue: 'best5', 'best6', or ('best6', b1(warps,replicas), route(warps,replicas), b2(warps,replicas)).
# attention: 'auto' or a best6 (family, setting) choice, e.g. ('layer',None), ('fa3',(16,True)).
# MoE: 'auto', 'v9nt1', 'v9nt2', 'raw17', 'humming_masked', 'dg_w8'.
# projections: 'auto' or {role: 'auto'/'custom'/'cublas'/('stream',config,splits)}.
# Roles: qkv_gate, o, pool_router, unpooler (unpooler_slice at tokens<=8).
# Reserved projections key: 'l2_prefetch': P7 recipe id. Absent/0 is disabled.
# Dense chain is a separate receipt-gated switch. It is deliberately false until
# the real-weight H200 paired lab lands; the lab key is 5279d3e3903a57b5.
# Decode M1-32 only; enable per row after its paired whole-layer H200 receipt.
# NT1 matches board v9; small-M seed means b68afb72 reject NT2 below M128.
# M128 NT1: best6 three-seed gate/mean; M256 Humming: de6a56fc + 1b80ffc0.
# Decode1 attention is best6's measured attn_main, not another tuner draw.
# Paired 9c83cdb: freeze the measured glue geometries, old glue at decode32
# sliding, public layer scheduling for every q8 row. Scale-bank choice awaits E:C.
BEST7_SETTINGS = {
    'other':       ('best6', 'auto', 'auto', 'auto'),
    (1,  1, 2048): (('best6', (16,16), (8,1), (8,16)), ('attn_main', None), 'v9nt1', 'auto'),
    (1,  1,    0): (('best6', (16,16), (8,1), (8,16)), ('attn_main', None), 'v9nt1', 'auto'),
    (16, 1, 2048): (('best6', (16,1), (8,1), (16,1)), 'auto', 'v9nt1', 'auto'),
    (16, 1,    0): (('best6', (16,1), (8,1), (16,1)), 'auto', 'v9nt1', 'auto'),
    (32, 1, 2048): ('best5', 'auto', 'v9nt1', 'auto'),
    (32, 1,    0): (('best6', (16,1), (8,1), (16,1)), 'auto', 'v9nt1', 'auto'),
    (1,  8, 2048): (('best6', (16,4), (4,1), (8,16)), ('layer', None), 'v9nt1', 'auto'),
    (1,  8,    0): (('best6', (16,4), (4,1), (8,16)), ('layer', None), 'v9nt1', 'auto'),
    (16, 8, 2048): (('best6', (16,1), (4,1), (8,1)), ('layer', None), 'v9nt1', 'auto'),
    (16, 8,    0): (('best6', (16,1), (4,1), (8,1)), ('layer', None), 'v9nt1', 'auto'),
    (32, 8, 2048): (('best6', (16,1), (4,1), (16,1)), ('layer', None), 'humming_masked', 'auto'),
    (32, 8,    0): (('best6', (16,1), (4,1), (16,1)), ('layer', None), 'humming_masked', 'auto'),
}

# P7 persistent o -> boundary -> pool/router chain. Keep the switch in the same
# shape table as the other dispatch decisions, but do not enable it from local
# timing. The implementation is admitted only after the paired H200 gate.
BEST7_DENSE_CHAIN = {
    'other': False,
    (1, 1, 2048): False, (1, 1, 0): False,
    (16, 1, 2048): False, (16, 1, 0): False,
    (32, 1, 2048): False, (32, 1, 0): False,
    (1, 8, 2048): False, (1, 8, 0): False,
    (16, 8, 2048): False, (16, 8, 0): False,
    (32, 8, 2048): False, (32, 8, 0): False,
}
