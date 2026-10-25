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
- v9 payload `8e107b90`: prefill job **3af06b40b5f944a6** (source `aa1511c8`); landed: all arms lose (results below).
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

## Landed takeover results

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

H200 v10 MoE check **4dd00540cb8c46df** (source `dc7b0c84`, payload `c6165434`) landed: all arms lose (results below).
Local v10 completed all 16 seeds and gates (`runs/screen/nt1_takeover_lab_v10.log`): WPUB/S11 ratios at
M1/8/16/32 1.00819/0.99755/0.99570/1.00033; ALIGN/S11 at M8/16/32 1.00877/1.01491/1.03486.
A:A mean absolute drift 0.92/0.92/1.46/3.54 us: no local claim of a win. There is no integration promotion.

Current artifacts: `smallm_var_cell.py` `b091e0d1`, `var.cu` / `var_kernel.cu.inc` generated by `build_var.py`,
`prof/lab/moe_nt1_timeline.py` `c6165434`, `validation.json` bound to that payload's hash before the run starts.
The experimental WPUB-M32 candidate/patch under `runs/screen/nt1_s11_wpub.*` are generated by `patch_s11.py`,
map {1:95,8:69,16:69,32:65605}, onto frozen S11 `acda6c62`; patch `b11d43c4`, candidate `bb60b792`,
standalone cell `fc474b84`. It passes patch applicability and cell admission, not whole-layer tests or H200
promotion. All those arms are now closed losses; the structural four-arm diagnostic below is current.


## Structural limit diagnostic

Owner direction: the logical ~2–3 TB/s stage rate does not identify HBM duty. `dev/timeline/README.md`
(`c6bd22e2`) counts compulsory unique logical bytes and includes PDL overlap, not measured DRAM transactions.
The new `build_limit.py` -> `prof/lab/moe_nt1_limit4.py` tests M8/16/32, 16 public weight/router seeds:

- A: exact frozen S11 `acda6c62` consumer69, original loads/MMA/synchronization.
- B: wrong-math **timing-only** bounded BF16 reinterpret. Same packed/scales TMA/shared loads, activation
  loads, 32 MMA instructions per tile and synchronization. Hopper SASS verifies 32 HMMA, zero HMUL2.
- C: weight loads only: same tile addresses/order, repeated 8-token-block requests, grid264, two-stage
  TMA ring and S11 shared reservation. No activation/MMA/epilogue/ready/finalizer work. Checksums prove all words read.
- D4/D8/D16: maximum-MLP streams of the same unique expert ranges; no shared reservation; 4/8/16 queued
  CTAs/SM. Same checksums. Bytes/counts distinguish unique cold reads from requested block-repeated reads.

Unprofiled AB/BA x6 plus independent A:A; separate three-sample CUPTI node durations and observer/off
ratios per arm and seed. All A gates and C/D checksums passed in a full local 16-seed run; B/C/D are never
candidates. Local timings only check the apparatus. `limit_b.cu`/`limit_c.cu` compile for sm90a, no spills.
H200 MoE check **360810b110a44d36** (host source `7fbab395`, payload `eac899e7`) landed; results below. Target remains >=4.0 TB/s unique logical MoE bytes for a correct kernel.

## Recovered decode-once result

H200 **e4ad355e013844f1** (`reuse_cell.py` da32c26a; payload d37757ca): one CTA owns a unique expert
weight tile, decodes one K group and reuses it over all token blocks. Block0 activations are cached in 56
registers + 8 KiB shared; remaining blocks stream group operands. Both ONCE/REDO: 127 registers, 93,448
shared bytes, 2 CTAs/SM, no spills. Static SASS proves ONCE HMUL2=128 vs REDO=512 and both HMMA=128
for four blocks. Correctness: 600 checks including fresh-input graph replays + six forced glue checks;
worst L2 0.00248, relative to INC <=0.000064.

| M | S11 us | ONCE/S11 | REDO/S11 | ONCE vs REDO us |
|---|---:|---:|---:|---:|
| 8 | 70.72 | 1.30219 | 1.13239 | +11.88 |
| 16 | 81.26 | 1.29940 | 1.26197 | +3.02 |
| 32 | 110.44 | 1.33385 | 1.46301 | -14.75 (16/16 wins) |

Re-decode has a measured ~15 us cost at M32 in this fixed-work contrast, but ownership and activation
streaming cost more; no integration win. `reuse_model.md` separates the ECAP ceiling from decode/MMA
instruction work. Trace kernels spill because of intrusive per-phase clocks; use untraced kernels for timing.

Direct-S11 H200 **3af06b40b5f944a6**: RND 1.05218/1.07873/1.04993, ROT 1.04762/1.04977/1.06942,
PIPE 1.10243/1.12751/1.09328 at M8/16/32; M1 ROT 1.04822. Every variant loses.
H200 **4dd00540cb8c46df**: WPUB 1.01531/1.01392/1.00381; ALIGN 1.04072/1.07442/1.04853;
M1 WPUB 1.06500. Every variant loses. All gates pass, 16 seeds each.

## Cluster metadata (parked under the structural priority)

`build_meta.py` restores consumer101 (69+META) onto exact S11; `meta_cell.py` cab198fa. It consumes
P7's existing cluster C8/NT256 producer ABI without changing its serial metadata epilogue. 54 end-to-end
fresh graph replays at M8/16/32 and global starts0/96/224 pass; worst L2 0.00241, relative INC 0.000063.
Hopper consumer101 uses 123 registers, no spills. `build_meta_lab.py` separates CL_OFF (producer no metadata),
CL_BASE (producer metadata, S11 consumer), and CL_META (producer metadata, consumer101).
Local 16-seed total ratios vs S11 1.04087/1.04288/1.02925; direct consumer ratios
0.99853/1.00108/1.00061. No H200 send or promotion; lower priority than the structural experiment.


## Structural results and WGMMA rebuild

**360810b110a44d36**, 16 seeds, one MoE node per graph, fixed identical routed fixture, read flush.
B is wrong-math timing only; C/D checksums pass. Rates count unique compulsory logical bytes, not DRAM
transactions. Ratios are per-seed paired unprofiled graph events; node times are separate CUPTI samples.

| M | A S11 node us / TB/s | B reinterpret us / paired ratio | C matched loads us / TB/s | D best unique stream us / TB/s |
|---|---:|---:|---:|---:|
| 8 | 64.886 / 3.067 | 59.956 / 0.92986 | 47.586 / 4.192 | 49.228 / 4.056 |
| 16 | 75.932 / 2.653 | 69.370 / 0.91301 | 53.740 / 3.752 | 50.194 / 4.043 |
| 32 | 104.695 / 2.075 | 85.216 / 0.82805 | 69.284 / 3.124 | 53.224 / 4.079 |

C paired ratios 0.74514/0.72301/0.66836; best D 0.77086/0.67396/0.53037 (best by mean node time;
M32 D8 event ratio0.52854). A:A mean absolute drift0.360/0.356/0.281us. A output L2~0.00238;
observer/off ratios near1. B saves4.93/6.56/19.48us, B->C leaves12.37/15.63/15.93us of
MMA/activations/synchronization/glue. C minus best D is -1.64/+3.55/+16.06us, including the changed
repeat-request schedule and omitted route/activation/epilogue work. These are mixed limits: M8's original
weight-only pipeline already exceeds4TB/s, and M32 also needs fewer repeated requests. No actual HBM
counter was collected, so the byte rate is not HBM utilization.

`build_wg.py`, `wg_kernel.cu.inc` -> `wg_cell.py` **2dee212a**: RS WGMMA64 rows x adaptive8/16/32
columns, one decode per expert, no K-split,4x fewer norm atomics. Original packed codes unchanged;
raw scale bytes reordered losslessly. Native168regs/16B local,3CTA/SM. Four arms K64 stages4/8,
grid264/396. H200 **00f4adc334084ba2** (`prof/lab/moe_nt1_wg.py` **5ed3dc1a**),16 seeds:

| M | S11 event / node us | W4G2 ratio | W4G4 ratio | W8G2 ratio | W8G4 ratio / node us / TB/s |
|---|---:|---:|---:|---:|---:|
| 8 | 70.589 / 65.890 | 2.28616 | 1.63425 | 2.20890 | 1.57690 / 107.728 / 1.876 |
| 16 | 81.175 / 76.124 | 2.08045 | 1.65896 | 2.02050 | 1.62035 / 127.048 / 1.593 |
| 32 | 109.995 / 104.928 | 1.99225 | 1.55038 | 1.95579 | 1.51718 / 163.116 / 1.338 |

All correctness gates pass; all performance arms lose. The narrow stages issue many copies/barriers and
carry only a small packed-weight queue. This tests one ownership/layout implementation, not an impossibility
result for decode-once or WGMMA.

`build_wg_perm.py` -> `wgp_cell.py` **9588df1e** fuses the NT1 intra-K64 permutation into router and
W13 BF16 stores; activation copies become16B. Physical[k]=logical[P(k)] with
`P(k)=(k&~63)|((k&6)*8+((k>>4)&3)*4+((k>>3)&1)*2+(k&1))`.
Both permutations and the raw scale bank have inverse checks. The router scatters these stores inside the timed
kernel; prepare does no activation work or weight expansion.

`build_wg_coarse.py` -> `wg2_cell.py` **652d2a3f** (K128), `wg4_cell.py` **3fc27752** (K256).
Stages4/8 and2/4 respectively: shared54,152/105,352B; native153-157regs, no stack/spills or WGMMA
serialization warnings. Bundled MMA descriptors are computed before the WGMMA fence; pairs of K64
fragments retain constant buffer indices and valid async register lifetimes. Native residency3/2CTAs/SM.
`check_wg2.py` and `check_wg4.py`: each300 math/replay checks plus6 forced high-stage host-glue checks,
worst L2 0.0024594, maximum relative INC0.0009174. High stages exceed local shared capacity, so only
their host ABI was forced through lower-stage stand-ins. Native high-stage correctness is an H200 gate.
The full producer+consumer local16-seed lab passes all four arms (`run_coarse_local.py`), without timing claims.

Coarse payload `prof/lab/moe_nt1_wg_coarse.py` **27ce07ef**, H200 MoE **b3f1c4fbc7164374**, landed:
K128S4/K128S8/K256S2/K256S4 vs S11 consumer69, paired router+MoE graphs,16 seeds, read flush,
separate CUPTI node samples and A:A. No integration promotion; S11's flag map stays unchanged.


Coarse H200 results: all four native gates pass (L2~0.00238/row~0.00246),16seeds, A:A absolute drift
0.381/0.307/0.203us. Wider/vector copies improve the first RS rebuild but still lose; no win to hand off.

| M | S11 event/node us | K128S4 r | K128S8 r | K256S2 r | K256S4 r | Best new node us/TB/s |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 71.109/66.056 | 1.39575 | 1.49292 | 1.37177 | 1.47871 | 93.168/2.168 |
| 16 | 81.636/76.872 | 1.37395 | 1.34335 | 1.34650 | 1.32194 | 103.638/1.953 |
| 32 | 110.879/105.898 | 1.11186 | 1.12973 | 1.11834 | 1.11690 | 117.440/1.840 |

The next independent-queue design (`build_wg_queue.py`) isolates packed-weight buffering from activation
stages: one contiguous4608B bulk TMA perN64/K128,4/8/12 raw stages plus3 activation stages.
This increases packed bytes in flight without duplicating each8KiB BF16 activation buffer per raw stage.
Prepare only permutes the raw codes and u8 scale bytes into contiguous K128 units; timed dequant is unchanged.
Both raw-bank halves are verified by exact inverse permutation.

Independent-queue cell `wgq_cell.py` **224cbbb1**, payload `prof/lab/moe_nt1_wg_queue.py` **f9c21230**.
Native Hopper compile151-153regs/no stack/spills or WGMMA serialization warnings. Dynamic shared45,992/64,456/
82,920B, nominal native residency3/3/2CTAs perSM. `check_wgq.py`:900 checks,9 forced ABI launches,
worst L2 0.0024594, relative INC0.0009174. All three depths execute natively onsm120 (TMA plus portable MMA),
without high-stage stand-ins. `check_wgq_mem.py`:36 changed-input graph replays under compute-sanitizer,
zero errors. Initial distributed mbarrier initialization failed locally; single issuing-thread initialization
plus its release fence and a CTA barrier corrected it before sending. `run_queue_local.py` completes16seeds
with every router+MoE output gate passing. The local profiler explicitly selects CUDA12.4 record9, while the
remote payload retains its already H200-verified record10 ABI. These local timings are not H200 evidence.
`queue_model.md` separates the experimental contrasts from unmeasured HBM transactions and documents lifetimes.
The quota-blocked MoE/prefill local waiters were canceled before acceptance. R-20261025-3 authorizes
attention import-time labs; the payload now synthesizes all weights with `weights=None` and was retested
locally over16seeds. Attention H200 **a2e4cf101c1f4b92** (`e860cbae` payload, `81b89ff3` wrapper) landed:
Q4 ratios1.56709/1.55927/1.25969, Q8 1.57919/1.56827/1.26651, Q12 1.98494/1.81304/1.48110;
all gates pass, no win. No reset-time duplicate remains.

## Current: why decode-once loses, then activation overlap

Integration base is now **S13 final958f1258**. Its small-M CUDA is byte-identical toS11 (`c023c946`),
so flags69 remain the stage control. `phase_findings.md` records the measured contrasts and limits.
`build_reuse_phase.py` -> `reusep_cell.py` **d1ec8fbe**: mode0 BASE reproduces original ONCE;
mode2 BUNDLE specializes block counts and reduces all blocks in parallel. Stage2 untraced kernels keep
2H200 CTAs/SM,127/128regs, zero stack/spills; the separate phase builds have8B local, not the old176B.
1320 local checks +6forced launches pass; local BUNDLE TRACE uses1 raw stage solely to fit shared capacity.

H200 attention **f58cc7a6a6694326** (`ebf9c15b` payload, `1faceb7e` wrapper),16seeds:
BUNDLE/S13=1.13216/1.16782/1.24438. Against BASE it saves12.215/10.935/10.639us (16/16 wins each),
but no arm beats S13. M32 critical-unit math58.7->49.2us, epi28.5->22.8, weight wait3.6->1.7;
grid barrier19.9->25.1. Trace/off1.032/1.017 bounds the observer tax, not exact untraced attribution.
BUNDLE W13-unit math grows0.90/3.64/6.41/9.16us at1/2/3/4blocks while weight wait stays~0.07us.
SASS proves one128HMUL2 decode per specialized path; growth therefore remains in additional activation
loads/address/control/MMAs, not a repeated decoder or long weight-ready waits.

The next variants issue per-thread extra-block activation copies before dequant (ACT, `build_reuse_act.py`),
and optionally stage decoded A once in the reduction workspace (DECODED, `build_reuse_pipe.py`). This keeps
8-way Ksplit,2 packed stages and queried2H200 CTAs/SM. Expanded values exist only transiently in timed shared
memory; prepare still holds raw packed weights. Native builds spill under the128-register bound, recorded
explicitly. Optimized local math uses1 raw stage because sm120 cannot fit the H200's shared footprint;
all edge routes and fresh-input replay tests must pass before sending. The all-block A workspace needs
barriers both before the epilogue overwrites it and after reduction before the next tile writes A.
The first local DECODED edge test exposed that overwrite race; it was fixed before any send.

No integration win or patch. The incumbent small-M map stays `{1:95,8:69,16:69,32:69}`.
