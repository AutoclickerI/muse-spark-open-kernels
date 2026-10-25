# P7 to P3a: cluster-route metadata experiment

`patch_route_meta.py` applies onto S11 final `acda6c62`, preparing per-M metadata
and arrival buffers and passing `data['route_meta']` into `_best6_call`'s small-M
consumer. Enable `state['cluster_route_meta_on']=True` in the experimental arm.
The production default remains off until a paired whole-layer win lands.

The producer is the existing `route_cl` with C=8, NT=256. It uses the
`MoERouteMeta` v1 ABI in `nt1_branches/route_meta.h` (736 int32 words including
padding, 711 meaningful field words). Counts and pair maps use local expert IDs
0..31; the header's local_start is the actual global expert range. The consumer
receives local `lids` and config.local_start=0, so it must not subtract the
header's global local_start from these maps. The arrival counter resets in the
same invocation. No host input-dependent control or timed clearing is needed.

The current producer still builds the final metadata in the last-arriving
token's rank-0 CTA. This is a serial epilogue within the cluster route kernel;
it should not be assumed to have removed the previous META bottleneck.
Measure producer+consumer against the exact same route geometry with metadata
disabled, as well as against S11's full layer. Decode-32 sliding changes route
geometry from its best5 route to cluster, so separate that change's effect.

P3a now provides `nt1_timeline/meta_cell.py` (d9b5cb13), restoring the direct metadata consumer as flags 101 against flags 69. P7 does not edit that active consumer. The new producer-glue route epilogue emits the identical v1 ABI without a separate cluster route launch. M=1's SLOT path already avoids much of route
reconstruction, so M=8/16/32 are the primary consumers to measure.

Existing producer validation: `test_meta_local.py` compares all 711 meaningful
words with `make_route_meta` at M=1,2,8,16,31,32, starts 0,96,224, eager plus six
fresh graph replays. A new end-to-end consumer needs its own replay and whole
layer validation before promotion.
