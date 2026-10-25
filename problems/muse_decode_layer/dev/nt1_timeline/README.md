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

## Phase schedule at M=16/32
M=32 critical CTA with 69 (c768a8e3, one seed): route 4.2, W13 36.1, W2-ready wait 11.6, W2 17.6, grid barrier 5.0 us
of a 76 us span. Every CTA holds a contiguous unit range, so all experts finish W13 together and W2 waits on the slowest
CTA of its expert plus the publish (the wait ends ~4 us after the last W13-end stamp). RND (flag 8192): two rounds
split at the expert boundary nearest half the W13 units; each CTA runs its share of the first experts' W13, then the
rest, and W2 in the same order, so the first experts' W2 covers the second round's publish. Traces now split per-tile
time (tile waits of warp 0 and warp 7, per-tile barrier, publish fence and barrier): TILE lines.
The prior v7 hash `1be7ce445ced8f66` has no ledger/receipt and is unmeasured.

Takeover admission: F_RND, F_ROT (16384: delayed rotating warp epilogues, shared arrival counts in place of the
per-tile CTA barrier), and F_PIPE (32768, requires F_RND: W13-A, W2-A, W13-B, W2-B) build and pass `test_cell.py`.
`sm_90a` static compile also passes. v9 uses the S11 flags as the directly paired control (M1 95, M8/16/32 69),
16 seeds, read flush, alternating order, with per-seed ratios and win counts. M1 also tests 16463 (ROT, drops NREG).
Local `run_local.py` and `screen.py` now read the flush buffer through an int64 view, matching the remote wrapper;
the byte sum had allocated an 8 GiB temporary and OOMed under concurrent GPU0 use. Local timings with other lanes
using GPU0 are not H200 evidence. Expanded checks include no local routes, one expert, and eight local experts.

- v8 payload `6b3d185c`: H200 MoE check **17512d86f3a74f27**, source `0d031529`, landed (see results below).
  R69/R8261/R16453/R24645 at M8/16/32 versus INC, 16 seeds, TILE publish/barrier diagnostics.
- v9 payload `8e107b90`: prefill job **3af06b40b5f944a6** (source `aa1511c8`); direct S11 pairing, adds PIPE and M1 ROT. Results pending.
- Local v8: 918 math checks + 17 forced Hopper glue checks, worst TASK L2 0.00239, maximum relative vs INC 0.000263.
  v9 including the extra edge cases: 1860 math checks + 24 forced glue checks, worst TASK L2 0.00239 and
  max relative vs INC 0.000333. Cell (`b58da13c`) worst relative vs INC 0.000347. Promotion requires H200 results and a whole-layer paired follow-up.

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

Local v9 (`runs/screen/nt1_takeover_lab_v9.log`, 16 seeds, GPU0 shared, **not H200 evidence**): directly versus 69,
ROT r=1.00798 / 0.99782 / 0.98143 at M8/16/32 (M32 -3.03 us, wins 15/16). RND r=1.01270 / 1.01890 / 1.02828;
PIPE r=1.03540 / 1.04143 / 1.05378. A:A at M32 has 2.41 us mean absolute drift, so these local measurements
only rank the next H200 questions. All cells passed the TASK gate (L2/row ~0.0024).

## Landed takeover results and pending experiments

H200 MoE check **17512d86f3a74f27** (16 seeds, read flush, router+MoE graphs, all gates pass):

| M | S11 69 us | RND 8261 us | ROT 16453 us | RND+ROT 24645 us |
|---|---:|---:|---:|---:|
| 8 | 70.93 | 74.99 | 73.99 | 76.32 |
| 16 | 81.58 | 88.31 | 85.78 | 89.97 |
| 32 | 110.81 | 117.27 | 118.73 | 119.92 |

All lose to S11: derived ratios RND 1.0572/1.0825/1.0583, ROT 1.0431/1.0515/1.0715, combination
1.0760/1.1028/1.0822 (these are ratios of the aggregate arms paired to INC; v9 directly pairs each to S11).
At M32 RND cuts the critical W2-ready wait 11.20 -> 2.11 us, but W13 grows 37.41 -> 41.09 and the grid
barrier 4.34 -> 12.66 us; node 79.95 -> 85.36. ROT cuts warp7's accumulated tile barrier 11.89 -> 0.91 us,
but node 81.34 vs 79.95 in the traced seed and 118.73 vs 110.81 across seeds. This overturns the local ranking.
Publish fence median is ~1 us, publish barrier ~0.10 us (ROT 0.85); latest publish follows last W13 end by
2.29 us (baseline), 1.01 (RND), 2.43 (ROT). No integration win.

v10 (`c6165434` payload; `b091e0d1` cell) keeps S11 as control, tests **WPUB** (65536: two W13 writer
warps independently fence and publish, readiness target doubled; removes the publish CTA barrier) and **ALIGN**
(131072 + RND: parallel split search and m-chunks sized for a round's units, avoiding the extra partial-block
reloads of splitting a full-pass chunk). It does not repeat the closed ROT/PIPE arms. Words: M1 65615
(WPUB, drops NREG), M8/16/32 65605 (WPUB), 73797 (WPUB+RND), 139333 (ALIGN+RND), 204869 (+WPUB).
Cell, 1404 edge/math checks, 16 forced glue checks, and sm90 static compilation pass. Old closed NT2/W16
instantiations were removed from the active lab build and admission suite; their evidence remains above.
`patch_s11.py` creates a candidate, standalone cell and unified patch onto frozen S11 `acda6c62`, with an
explicit complete flag map; its WPUB experiment patch passes `git apply --check`. It is not a promotion.

H200 v10 MoE check **4dd00540cb8c46df** (source `dc7b0c84`, payload `c6165434`) accepted; result pending.
Local v10 completed all 16 seeds and gates (`runs/screen/nt1_takeover_lab_v10.log`): WPUB/S11 ratios at
M1/8/16/32 1.00819/0.99755/0.99570/1.00033; ALIGN/S11 at M8/16/32 1.00877/1.01491/1.03486.
A:A mean absolute drift 0.92/0.92/1.46/3.54 us: no local claim of a win. There is no integration promotion.

Current artifacts: `smallm_var_cell.py` `b091e0d1`, `var.cu` / `var_kernel.cu.inc` generated by `build_var.py`,
`prof/lab/moe_nt1_timeline.py` `c6165434`, `validation.json` bound to that payload's hash before the run starts.
The experimental WPUB-M32 candidate/patch under `runs/screen/nt1_s11_wpub.*` are generated by `patch_s11.py`,
map {1:95,8:69,16:69,32:65605}, onto frozen S11 `acda6c62`; patch `b11d43c4`, candidate `bb60b792`,
standalone cell `fc474b84`. It passes patch applicability and cell admission, not whole-layer tests or H200
promotion. Next: read **3af06b40b5f944a6** (v9) and **4dd00540cb8c46df** (v10); select any robust winner's
per-M map, generate its patch, run normal and forced local_eval, then give integration the paired whole-layer
measurement. If both lose, keep S11's map and target a different multi-block storage/scheduling mechanism;
repeating RND, ROT or PIPE unchanged is closed.
