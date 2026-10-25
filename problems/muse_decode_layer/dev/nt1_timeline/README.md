# Small-M NT1 (M=1-32): timeline and flag variants

Lane notes for PLAN P3a. Incumbent: the admitted one-LOP3 NT1 (`inc.cu`, identical to
`nt1_branches/nt1_lop3_cell.py`). Everything here reads the same recoded bank (4-bit codes, one u8 scale per
16 weights, TILE_BYTES 36864); nothing widens or dequantizes scales in prepare.

## Files
- `var_kernel.cu.inc` + `build_var.py` -> `var.cu`: `moe_var<1, F>`, the incumbent kernel with compile-time flags
  (F=0 is the incumbent's math): Z=1 (norm slot zeroed in W13 tile 0, no C_ZERO wait), SLOT=2 (M=1: dispatch from
  the 8 top-k slots, rank = popcount; no counts/scan/lists/binary search; first TMA issued by warp 0 before the CTA
  barrier), ZOUT=4 (W2 stores z = v(1+w); finalizer drops post-norm weight loads), RSQ=8 (finalizer: lanes 0-7
  compute each pair's scale once, shuffle), NREG=16 (W2 norm partials in registers across a CTA's run of tiles for
  one (expert, block); one reduce + atomic per run), PDL=64 (programmatic dependent launch; setup before `griddepcontrol.wait`), LAST=128 (last CTA to
  finish W2 runs the whole finalizer; no grid-wide release), TRACE=256 (globaltimer stamps, layout in the kernel).
- `smallm_var_cell.py` (`build_cell.py`): decode-lane drop-in. `smallm_bind(s4_nt1_bank, {M: flags})` in prepare,
  `smallm_experts(data, s4_nt1_bank, flags)` replaces `nt1_call(data, s4_nt1_bank, 'lop3', 2, 264)`.
  `test_cell.py` checks it against INC.
- Lab payload `prof/lab/moe_nt1_timeline.py` (`prof/lab/build_moe_nt1_timeline.py`): M=1 timeline (CUPTI node
  records optional; marker kernels, EMPTY and STREAM controls, globaltimer resolution) in E (MoE alone) and R
  (router + MoE) context, then every arm paired AB/BA vs INC at M=1/8/16/32 over 16 seeds, TASK.md gate per cell.
- `check.py` -> `validation.json`: every arm's math (TASK.md gate and
  <2e-3 relative to INC) on real and edge routes (none/one/all-8 local, half remote, uniform, zero input), traced
  builds, fresh graph replays, forced Hopper glue. `run_local.py` runs the payload as the wrapper does.
- `table_kill/`: joint E2M1 x E4M3 table decoder SASS count (stopped: 90.6 / 34.3 vs LOP3 26.06 per 16 weights).
- `screen.py [M=..] [SEEDS=..] flags...`: local paired screen of router+MoE arms plus their traced critical paths.

## Handoff (decode lane)
`smallm_var_cell.py` builds flags 95 and 69 (SHA-256 in the lane report; `test_cell.py` passes); map
`{1: 95, 8: 69, 16: 69, 32: 69}` (H200 cf1de160 and eb9e0382, router+MoE graph, 16 seeds each). Integration in S7:
`smallm_bind(state['s4_nt1_bank'], FLAGS)` at the end of `_s4_prepare_banks`; in `_best6_call`, `nt1_lop3` ->
`smallm_experts(data, state['s4_nt1_bank'], FLAGS[m])`. PDL needs nothing else: S7 launches only `torch.empty`
between the router (both routers trigger at start) and the MoE on every nt1_lop3 row.

## Phase schedule at M=16/32 (in progress)
M=32 critical CTA with 69 (c768a8e3, one seed): route 4.2, W13 36.1, W2-ready wait 11.6, W2 17.6, grid barrier 5.0 us
of a 76 us span. Every CTA holds a contiguous unit range, so all experts finish W13 together and W2 waits on the slowest
CTA of its expert plus the publish (the wait ends ~4 us after the last W13-end stamp). RND (flag 8192): two rounds
split at the expert boundary nearest half the W13 units; each CTA runs its share of the first experts' W13, then the
rest, and W2 in the same order, so the first experts' W2 covers the second round's publish. Traces now split per-tile
time (tile waits of warp 0 and warp 7, per-tile barrier, publish fence and barrier): TILE lines.
H200 v7 `1be7ce445ced8f66` (queued): 69 vs 69+RND at M=8/16/32, M=16/32 critical paths of both.

## H200 accounting (849e4017, M=1, INC, read flush)
Event 36.21 = CUPTI node 31.81 + 4.40 event framing. Node = launch 1.0 + CTA span 32.2 + drain 1.3; the critical
CTA's phases cover the span (waits: norm zero 4.05, W2 ready 6.42, grid barrier 2.64). All flags but META: span
27.4, node 28.1; vs a pure stream over the same 56.6 MB (node 16.7): route 1.5, first tile 3.7, W2-ready wait 4.7,
finalizer ~2.4 left. zero_() vs read flush on the H200: +1.8 / +4.9 / +2.7 / +1.9 us (M=1/8/16/32), not the local
+5.5-15.7.

## Judge runs (prefill host)
- `28fc0380d4ca4c5f`: v1, dirty `zero_()` flush. `849e40173ebd4bfc` (lab sha16 `eb5c766c7d2fd930`): v2, read flush,
  M=1 timeline, all flags at M=1/8/16/32. Both landed; same ranking.
- `05e21862a4044642` (job cf1de1607d394778): v3, `lab_cold_l2()`; R95 at M=1, R93/R77/R69 at M=8/16/32; ECAP (INC
  with each local expert capped at 8 tokens: the most decode-once could save) at M=16/32; M=32 timeline with SKEW
  lines. Local `runs/screen/nt1_timeline_lab_local_v3.log` (ECAP -2.3 / -8.5 us at M=16/32 on sm_120).
- `6c5fb6083f17a5e6` (job eb9e038277684174): v4, R1119 (95 + DYN) vs R95: DYN loses on the H200 (-1.78 vs -5.68)
  although it won locally (-5.33 vs -2.67); W13 slows by ~4 us per CTA. 69 confirmed at M=8/16/32.
- `15e7735184c220f2` (job daf0c1aa0bd24eeb) and `ce29e19cbb20c8f5` (job c768a8e361f54977): v5/v6, NT2 and W16 lose
  (see Closed recipes).

## Closed recipes
- META (router-written `MoERouteMeta`, flag 32, deleted): +2.9 to +3.3 us at every M on the H200; the router's serial
  metadata epilogue costs more than the per-CTA route rebuild it replaces.
- LAST (last CTA finalizes): +0.11 alone, +0.36 with SLOT (M=1).
- DYN (M=1 units grabbed from a global counter, deleted): removes the W2-ready wait but slows every CTA's W13 on the
  H200 (+3.9 us vs 95, eb9e0382). Static expert grouping (deleted) gained nothing locally.
- Router-epilogue L2 prefetch of the MoE's first tiles (M=1, local): router node 4.1 -> 51.2 us. A single CTA's
  bulk prefetches run at one SM's TMA rate and block its exit.
- Joint E2M1 x E4M3 table decoder: see `table_kill/`.
- Decode once in registers for multi-block experts (M=16/32; flags 2048 NT2, 4096 W16): vs INC, NT2 (16-token blocks,
  203 regs, 1 CTA/SM, 4 stages) +11.8 / +7.9 us and W16 (16 K-split warps, 123 regs, 1 CTA/SM) +19.0 / +17.2 us,
  while 69 is -4.7 / -4.5 (c768a8e3, 16 seeds). NT2 cuts W13 units 37% but a 2-block unit costs 1.84x an NT1 unit
  per SM, so the multi-block price is per-tile latency at 16 warps/SM, not decode issue; 16-token register
  operands do not fit 2 CTAs/SM. ECAP (each expert capped at 8 tokens) had bounded it at -9.3 / -34.5 us.
