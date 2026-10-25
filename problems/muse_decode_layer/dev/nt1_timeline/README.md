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
`smallm_var_cell.py` instantiates flags 69, 77, 93, 95, 1119 (SHA-256 in the lane report; `test_cell.py` passes).
H200 map (router+MoE graph, cf1de160, 16 seeds): `{1: 95, 8: 69, 16: 69, 32: 69}`; switch M=1 to 1119 (95 + DYN)
if 6c5fb608 confirms the local -2.7 us. Integration in S7: `smallm_bind(state['s4_nt1_bank'], FLAGS)` at the end of
`_s4_prepare_banks`; in `_best6_call`, `nt1_lop3` -> `smallm_experts(data, state['s4_nt1_bank'], FLAGS[m])`. PDL
needs nothing else: S7 launches only `torch.empty` between the router (both routers trigger at start) and the MoE
on every nt1_lop3 row.

## Next: decode-once for multi-block experts (M=16/32)
ECAP (INC with every local expert capped at 8 tokens) bounds it at -9.3 / -34.5 us (cf1de160): each extra 8-token
block costs ~7.4 us at M=32 though its tiles are L2 re-reads. First test: `moe_var<2, F>` (16-token blocks) at 1
CTA/SM with 4 stages (Bf[2] needs ~128 registers; NT4 spilled at 2 CTAs/SM in b68afb72).

## DYN (flag 1024, M=1 with SLOT)
The producer grabs 1-4 units at a time from a global counter (all W13 units, then all W2) and queues each unit id in
its stage; consumers read the id one stage ahead, so operand reloads and W2-ready waits still overlap the copy. It
balances W13 completion across CTAs (W2-ready wait on the critical CTA 14.2 -> 7.2 us locally). At M>=8 grabbed
chunks split a multi-block expert's tile reuse across CTAs: +3 to +19 us locally, so DYN is M=1 only. A static
expert grouping (early experts' W13 first) removed the wait but gained nothing (deleted).

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
- `6c5fb6083f17a5e6` (queued): v4, R1119 (95 + DYN) vs R95 at M=1 with both critical paths; R93/R77/R69 again.
  Local `runs/screen/nt1_timeline_lab_local_v4.log` (R1119 -5.33 vs R95 -2.67).

## Closed recipes
- META (router-written `MoERouteMeta`, flag 32, deleted): +2.9 to +3.3 us at every M on the H200; the router's serial
  metadata epilogue costs more than the per-CTA route rebuild it replaces.
- LAST (last CTA finalizes): +0.11 alone, +0.36 with SLOT (M=1).
- Router-epilogue L2 prefetch of the MoE's first tiles (M=1, local): router node 4.1 -> 51.2 us. A single CTA's
  bulk prefetches run at one SM's TMA rate and block its exit.
- Joint E2M1 x E4M3 table decoder: see `table_kill/`.
