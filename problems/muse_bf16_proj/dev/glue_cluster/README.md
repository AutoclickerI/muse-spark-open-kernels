# Cluster glue (P7): boundary and route on a thread-block cluster per token

S6's Triton glue gives each token one CTA (or replicas that each read the whole row), so at S split
partials one SM's L2 bandwidth bounds it (route at M=32 reads 16 x 4608 x 4 B per token: 6.5 us).

- `glue_cl.cu`: `boundary_cl<C, NT, ROUNDS>` and `route_cl<C, NT>`; C CTAs per token, each reads 1/C of
  the row, norm partials pushed into every CTA's shared memory (DSM) and one cluster barrier (ROUNDS 2:
  exact two-pass |stream'|^2). Route: per-split hi + lo logits in S6's order, top-8 by redux max of the
  order-preserving key with ties to the lower id (S6's key order). PDL release at entry, weight vectors
  before `griddepcontrol.wait`.
- `lab_body.py` + `build.py` -> `prof/lab/glue_cluster.py`: S6 producer (kind and splits per role/M) ->
  glue -> PDL consumer, S6 Triton glue at its BEST7 geometry vs cluster variants, t = 1, 8, 16, 32, 128, 256.
  H200 `1365a29092ac4a7f`: at t = 1 / 8 / 16 / 32 boundary ('cl', 8, 256, 1) b1 -1.01 / -0.53 / -1.14 / -0.53 us,
  b2 -1.42 / -1.04 / -1.71 / -0.61, route ('cl', 8, 256, 0) -0.26 / -0.38 / -0.37 / -1.41 (per layer -2.7 / -1.9 /
  -3.2 / -2.6 us); every cluster shape loses at t = 128 / 256 (1.02-1.28x). Route ids exact, boundary <= 3.5e-5.
- `patch_glue_cl.py`: ('cl', C, NT, r) glue entries; ROWS = the H200 choice on the eight t <= 32 rows (decode 32
  sliding moves from best5 to best6 + cluster). `cand_r3_best7_s7_p7c.py` = S7 + early release + vdsk + this:
  13/13 local tests, 23 static checks.
- MoERouteMeta (P3a item 3, `../../../muse_decode_layer/dev/nt1_branches/route_meta.h`): `route_cl(..., meta,
  arrive, ...)` with NT 256 and t <= 32 writes the v1 metadata from the last-arriving token's rank-0 CTA
  (arrival counter, release/acquire, reset by that CTA; no CTA waits). `test_meta_local.py`: all 711 field
  words equal `make_route_meta` at t = 1..32, three EP starts, eager and 6 graph replays. Candidates pass
  meta = 0 until the small-M MoE consumes it (its variant A without the separate metadata launch).
