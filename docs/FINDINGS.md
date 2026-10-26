# Findings

Current evidence for G1/G2. **Measured** = H200 receipt; **static** = code/SASS/spec;
**model** = stated assumptions; **local** = sm_120, not H200 timing; **official** = organizer facts.
Job IDs resolve to `runs/<problem>/*_<job>.json` (prefixes accepted); lab output:
`tools/h200_lab.py read <job>`. Source hashes identify candidate bytes, not git commits.
A negative closes only its exact recipe. Work in flight belongs to PLAN.

## Boards

The organizers publish overall ranks only. Our fit (`tools/board_model.py`, snapshots `runs/boards/`;
model, not the organizer formula):

`score = max(decode, 0.351 + 0.312 * MoE + 0.268 * projection + 0.105 * attention)`

Missing components count 1.0x; prefill is ignored; `solved` counts presence only.

| Snapshot | Teams | Kendall tau all / top 300 | Exact ranks | Top 50 |
|---|---:|---|---|---|
| 20261023T182817Z, fit | 567 | 0.9549 / 0.9699 | 102; mean abs error 9.99 | 50/50 exact |
| 20261025T001018Z, no refit | 997 | 0.9637 / 0.9874 | 262; mean abs error 13.8 | 48/50 exact, 50/50 overlap |

Alternatives' tau all / top 300: arithmetic 0.9133 / 0.8248, geometric 0.9030 / 0.7870,
harmonic 0.8928 / 0.7592, rank sums/Borda <= 0.7722 / 0.8054; solved-first did not help.
Every top-50 team's fitted score is its decode-layer speedup; the leader's component composite is
1.87 vs 2.285x. Components matter through the layer.
Leaders exceed the practical model (75% of 4.8 TB/s + launch costs): snapshot 20261025T014433Z,
MoE 2.52x vs 2.35x model ceiling, layer 2.285x vs 2.25x. Projection leader 1.61x is 1.255x
its floor ( ~3.82 TB/s). These ceilings reflect model assumptions.

## Judge, quotas and loop

### The judge's H200

| Measurement | Value | Receipt |
|---|---|---|
| Endpoint state, not sampled clocks | 132 SMs, SM 1980 MHz, memory 3201 MHz, 700 W limit, ~130 W idle | `7fb3e930`, `849bd4b6` |
| Subtracted graph fixed cost | ~4.2 us; `replay_overhead` 4.26 us | `7fb3e930`, `0a43baec` |
| Cold dense reads, median of 16 minus empty graph 3.10 us | 8.4 / 16.8 / 37.7 / 67.1 / 75.5 MB: 4.11 / 5.95 / 10.88 / 16.88 / 18.85 us; fit **2.36 us + bytes / 4.58 TB/s marginal** | `7b6c96b5` |
| Stream recipe behind that fit | Grid-stride 16 B `ld.nc`, 132-1056 CTAs, 256-1024 threads, 4-8 loads in flight; TMA rings/slices never top 6 | `7b6c96b5` |
| 256 MiB plain cold read | 74.33 us, 3.61 TB/s; one recipe, not the roof | `4233660b` |
| Request concurrency, 64 MiB, 256 threads, 66/132/264 CTAs | LDG128 U1-4: 0.64-1.61 / 1.16-2.49 / 1.95-3.01 TB/s; cp.async U1-8: 0.55-1.89 / 1.01-2.64 / 1.72-3.15; bulk U1-8: 0.44-1.77 / 0.85-2.46 / 1.46-2.88; 528-CTA stream 3.80 | `849bd4b6` |
| Warm page-1 gather, 128 B rows, no math | 132 / 528 / 1056 CTAs: 0.459 / 1.615 / 2.558 TB/s | `849bd4b6` |
| Dependent pair, PDL versus normal | Saves 1.06 us in the probe | `849bd4b6` |
| Warm L2, 4/16/32 MiB | 7.3-7.8 TB/s; eviction hints < 5%, no 32 MiB cliff | `849bd4b6` |
| BF16 MMA vs WGMMA SS, weights M64/K64 | N8 504 vs 200, N16 545 vs 363, N32 590 vs 608 TFLOP/s; implementation rates, not peaks | `849bd4b6` |

The cold-read intercept includes launch/flush/ramp/tail: 1.2x floor at 67-75 MB, 1.4x at
38 MB, 1.7-2.35x at8-17 MB. It is not a universal hideable ramp (`5cefdbc4`).

### Timing protocol and prepare

Source: `msaroufim/openkernelchallenge-audit`, `harness/worker.py`, commit `0a68efe`, hash-matched to receipts' `evaluator.files`.
Two untimed eager calls, side-stream capture, one untimed replay; timed replays copy fresh inputs,
read-flush L2, spin ~1 us, replay, subtract calibrated replay cost. Extra streams fork/join inside
capture; `allow_tf32=False`. Profiles are one eager call, not graph medians.
Per-load ratio is `(46*SGLang sliding + 16*SGLang full)/(46*ours sliding + 16*ours full)`;
score is the six-load geomean (reproduces `0a43baec` to 1e-5).

Prepare runs once on that run's weights. Tensors in plain containers are cloned behind random pads,
up to 16 copies (decode 1-3, MoE 4-16, proj 10-16); cases rotate copies. Other objects are shared:
cached pointers stay valid, address tricks do not. Every run changes weights/routes.
Cold native compilation needs `CCACHE_DISABLE=1`; a fresh extension directory can still hit image ccache
(92.2 s locally). Parallel LT/attention compilation reduced prepare to 133.5 s (`43049b80`).
Prepare cap 300 s; total run 15 minutes. Quotas: 20 checks + 10 scored runs per problem/day,
reset 00: 00 UTC = 09: 00 KST; accepted-but-unqueued sends count (`053c794fc0aba59a` detached log).
Static validation failures, including compile errors, return 112.

Single scored runs cannot resolve small changes: starter CI +-1.5%; unchanged paths move ~4.5%
per shape (`0a43baec`, `4e212ec2`, `5c7323ea`). Paired protocol: identical inputs/routes,
12 public weight/router seeds x 3 fixtures, cold 128-256 MiB read flush, alternating AB/BA + A:A,
FP32 oracle on supplied IDs; A:A < 1-1.5 us (`58ae2e2c`, `9c83cdb8`, `383ccc1f`).

### Judge queue

95 runs to 2026-10-24 03: 40: starts median 399 s apart, runtime median 125 s; correlation with
previous duration 0.06 over 94 pairs. Longer packed labs buy evidence up to the run cap.
Queue estimate = position x ~6.6 min; under load it stretches (1,709 jobs/64 GPUs, three runs
in 10-24 13: 35-15: 02). Withdraw does not cancel queued execution (`07d07d96`). These are
receipt-timestamp observations, not a current service guarantee.

## Whole layer and integration

### Structure (static)

62 layers = 46 sliding (2048) + 16 full, all MoE in this problem. Per TP8/EP8 GPU/layer:
qkv_gate 37.7 MB, O 16.8, pooler 67.1, unpooler 67.1 (8.4 MB slice only when call T <= 8),
router8.4 MB FP32; 32 local experts, 28.3 MB each. Sources: `sim/muse.py`, official TASKs.
Order: qkv → QK norm/RoPE/KV write → attention → O → residual/RMS boundary1 →
pooler + FP32 router → top-8 → W13/SwiGLU → W2/post-norm/weighted sum → unpooler → boundary 2.
8192-wide residual norms and the 4096-wide pooled norm are true joins; router needs h, not
pooler output; expert addresses exist only after top-8.

### SGLang's decode layer (the 1.00x)

`b90f4b3c`, us at batch 1 / 16 / 32: decode sliding 169.5 / 252.2 / 253.3,
full 179.7 / 439.0 / 614.1; verify sliding 260.5 / 556.0 / 757.5,
full 266.0 / 721.6 / 1112.8. 19-34% of floor; fit 1.2 us + bytes / 1.25 TB/s.
Calibration profiles (`runs/calibration/profiles/`): decode-1 sliding 180 us = Marlin 45 +
cuBLAS/router 55 + FA3 10 + ~70 glue over ~40 kernels; decode-32 full 659 = Marlin 149 +
FA3 375 + GEMM 93; verify-32 sliding 798, Marlin 604.

### Our kernels on the H200

Incumbent: S15 **60254883** (`problems/muse_decode_layer/cand_r3_best7_s15.py`, `dev/s15/`), scored
1.800345x (`9c5fc18d`); paired S15/S14 1.0549 / 1.0510 / 1.0276 / 0.9895 / 1.0100 / 1.0227 at d1 / d16 / d32 /
v1 / v16 / v32, geomean 1.026 (`281b1383`). S14 final f1c77ce6 scored 1.742759x (`9b94c126`).
S14 trial hosts scored 1.7174 / 1.7530 / 1.7324x (`0441f9fc`,
`740c63af`, `2bbb626c`); only the last host is exact f1c77ce6.

Reset integration decisions (2026-10-26; stage nominations still require the combined S16 pair):

| Recipe | Result vs control / A:A | Mechanism | Leaves open (and decision) | Receipt |
|---|---|---|---|---|
| Exact S14 final f1c77ce6 scored |1.742759x SGLang; unpaired, no A:A | Retained S14 portfolio | Retired by S15 | `9b94c126d450448a` |
| Exact S15 60254883 scored |1.800345x SGLang; unpaired, no A:A | NF projection rows + last-CTA sliding d1 + register-fold full verify | Frozen incumbent (paired geomean 1.026 over S14, `281b1383`) | `9c5fc18d302c4d40` |
| S15 leave-one-out,16 seeds | S14/S15 at d1/d16/d32/v1/v16/v32=1.054865/1.050962/1.027606/0.989473/1.009972/1.022677; descriptive six-load R1.025676, A:A0.996805 | NF projection rows, last-CTA d1 sliding and register-fold full verify compose positively overall | Owner froze exact60254883. Legacy private-equivalence failures do not veto TASK, but missing seed covariance/per-arm TASK means no new held-out CI can be reconstructed | `281b1383359c4fe2` |
| Compact early sliding history on S14 | E4/OLD4=0.997414, SE0.001244; A:A range0.988238-1.095457. E2/E1/L4=1.016159/1.040193/1.020247 | Gather plus V transpose before dependency wait, smaller shared history | No independent promotion; the small point estimate loses to the selected C128 estimate and has large uncertainty. Other history ownership stays open | `402819c28279466d` |
| Shared-merge global attention on S11 host, versus OLD last/partial | Best new/OLD at d1S/d16S/d32S/d1F:0.995906/1.009411/1.001125/1.121209; full A:A radii0.056417/0.047526/0.018671/0.035314 | Last-CTA shared merge weights and fixed split waves | Original captured paths have no TASK gates: standalone winner unmeasured. C128 d1S is experimental in S16; gated exact-kernel repeat bb7239af and independent composed TASK pair must establish it. Full C128 timing loses | `ae6cd27877cc4ee2` |
| Small-M PAIR epilogue nomination | M8/16/32 PAIR/S14=0.97493/0.97262/0.97932, SE0.00235/0.00357/0.00280; A:A absolute0.415/0.235/0.225us on71.033/81.533/110.753us controls | Two contiguous float2 planes/two writers plus per-run norm accumulation | Select PAIR M8/16/32 and NORM M1 for composed R confirmation; M1's small gain remains uncertain. ONCE variants lose1.235-1.331x at M8-32; new transport remains open | `d650c04ead114119` |
| Native verify-sliding DSM cluster nomination | B1 C16/control0.719269, SE0.002772, A:A radius0.059278; B16 C8/control0.849109, SE0.003406, A:A0.033439. B32 best C4 loses1.073812 | Distributed output rows across cluster CTAs, avoiding last-CTA serialization | Nominate B1/B16 sliding verify to S16; retain B32. New stage result is not yet an S15 whole-layer gain | `b05713ecf6574ad9` |
| Full-verify softmax batch on N64/stage6 | B16/B32 mixed25%=1.05012/1.04610; mixed50%=1.08436/1.07352; TC25=1.03473/1.03452; ping=1.15929/1.17091; best gap=1.00004/1.00126. A:A radii0.00438/0.00392 | Polynomial/MUFU mix, TC row sums, two-WG scheduling | No arm nominated; these recipes lose or overlap A:A. S15 N128/stage4 softmax and other schedules remain open | `457f3c73b3214534` |
| Heavy direct-copy launch and S3 policies | Ordinary direct/control1.1965/1.1381 at M128/256; other policies1.1427-1.2992; max A:A3.120us on182.90/256.67us controls; all TASK PASS | Direct activation copies, cluster/ordinary/serial launches, optional S3 | No heavy arm admitted; closes these transport/launch policies, leaves cheaper reuse and ownership open | `8792099636654bac` |
| PDL/boundary reset archive | Unmeasured: official host passes, but no LAB_BEGIN/BD_BATCH_XZ85 survives the16000-character receipt tail | No LAB marker survives; cause not established | No boundary/PDL arm admitted from this receipt; requires a surviving paired archive | `4c86853a953b4eca` |
| S15e producer-release composition, repaired entry point | Six-load S15/E R of paired load means1.004957; A:A R0.999668. Attention/MoE off-over-full R1.001668/1.001811; projection0.999729. Eight seeds, complete gates; seed covariance discarded, no exact CI | PDL producer release and consumer launch/wait edges, unchanged arithmetic | Attention/MoE selected for a future held-out composition; projection omitted. No standalone promotion from aggregate means | `f88fda4aac144374` |
| S15n nearby range dispatch,65 cases | Nearby/scored speedup by six loads: S15 .9547/.8651/.8184/.8119/.7327/.7263 → S15n1.0292/1.0142/.8884/.8682/1.0050/.9360. Scored identical-dispatch seed ratios span .97794-1.03416; no independent A:A arm | Range-based projection/MoE/attention plans remove nearby shape cliffs while retaining scored dispatch | Timing criterion passes; native correctness unproven (finite-only lab). Guarded9cc90d2e has native TASK audit sender1096763 queued; no final promotion yet | `a793c99eb9d447df` |
| S15L owned-library composition, repaired entry point | Full six-load R0.998959, A:A R0.998656;768 TASK checks pass. O off/full at v16/v32=1.004843/1.001366, pool128 at v16=1.001877; unpool1280.996584, v1full0.992091, v32slide0.993620 | NF O128/256, dependent VDSK pool/unpool, VQ8 replacements | Nominate O and pool128 only. Full L loses; actual changed-load unpool/attention replacements lose. Seed covariance missing: all aggregates are selection evidence, not promotion CI | `f22020d25d234be6` |
| First S16 composition 2e26db54 scored |1.796154x SGLang, unpaired; no promotion CI yet | C128 d1S, DSM v1S/v16S, NORM M1, PAIR M8/16/32 on exact S15 | Correct scored source; freeze waits on held-out six-load confirmation. Unpaired S15/S16 scores do not rank this small difference | `7139485dbeef4e0d` |
| ILP replaces PAIR in the small-M MoE | ILP/PAIR M8/16/32=1.01094/1.02063/1.01190;32 fresh seeds. A:A absolute0.488/0.377/0.364us, max1.424/0.928/1.648us | Alternate MMA accumulator chains lose to the admitted float2 epilogue despite beating S14 | Retain PAIR; close exact ILP replacement, leave combined instruction/epilogue edits open | `1b135b70835f4634` |
| Boundary/PDL archive retry | Unmeasured: LAB_ERROR combined archive exceeds lab log cap; no paired ratios or A:A available | Broad attribution archive cannot be emitted intact | No promotion or recipe closure; P7 pursues the larger MoE shell | `ce5b991a8a984a03` |

Original S15 admission wrapper49ce378b was rejected422 before a job existed: its packed source had no
explicit top-level `kernel_fn`. Entry-only repair2a4d0dde passed normal/forced full-wrapper screens and ran
as281b1383. The old per-load/host-equivalence veto was superseded by the score policy in97e98f3;
exact60254883 is owner-frozen. That legacy receipt lacks raw seed covariance and per-arm TASK checks,
so its geomean is descriptive rather than a new held-out confidence claim. S16 records both explicitly.

Incumbent pieces and their admission evidence (r > 1 means removing the piece is slower):

| Piece retained | Paired evidence | Job |
|---|---|---|
| Lossless uint8 scale recode; one-LOP3 NT1 | Recode M8/16/32/128 0.914 / 0.908 / 0.897 / 0.880 vs raw; one-LOP3 M1/8/16/32 42.7 / 73.8 / 79.2 / 113.6 us vs recoded 43.1 / 74.5 / 81.2 / 117.2 | `58ae2e2c`, `dc3fea0a` |
| Small-M map {1: 95, 8/16/32: 69} and BF16 front | Reverting map r 1.054 / 1.028 / 1.017 / 1.037 at d1/d16/d32/v1; removing front 1.032 at d16 | `ff3b9aca` |
| Balanced GEMVs, pinned small-M custom paths | Off +1.7 / +1.9 / +3.4 / +1.9% at d1/d16/d32/v1 | `863f502c` |
| Gate split, late glue | Off +1.4% at d1; late-glue off +0.7% at v32 | `863f502c` |
| LI_swap N_full heavy loop | Old N_full off +1.2 / +1.1% at v16/v32 | `863f502c` |
| Full verify vq8 page ring | Off +4.4 / +8.2% at v16/v32 | `863f502c` |
| Coded qkv/pool_router at t=1; split finalizer F4 | Coded off +3.9% at d1 (A:A +1.1%); F4 off +0.60 / +0.73% at v16/v32 | `937801fe` |
| Direct-partials attention at decode 1; PERMW256 | Attention off 1.0377; PERMW256 off pooled 1.0038, worst A:A 1.0033 | `cb38a050` |
| Light experts N16G4 at M128/256, PERMW both | Off +6.2 / +2.8%, +2.3 / +1.8%, +1.7 / +1.7% at v16/v32 in three runs | `0441f9fc`, `740c63af`, `2bbb626c` |
| bal2 pool/router at t=32 | Off pooled 1.0104, worst A:A radius 0.0054; neutral in last run | `0441f9fc`, `740c63af`, `2bbb626c` |
| Immutable glue preloads at decode-1 full | Held-out full 0.987 (A:A 0.995), sliding 1.001; full only | `1a880faf` |

S14 promotion used pooled controls, not a claim the last run alone was clean: latest light-off
verify 1 1.006641 exceeded .003130 A:A radius; pooled verify 1 1.001118 and decode 1/16/32
.999650/1.003959/1.001425 were within worst observed envelopes (`0441f9fc`, `740c63af`, `2bbb626c`).

Exact S14 host `2bbb626c`: per-load sliding/full call medians (us), copied from its profile report:

| Load | Ours S / F | Paired SGLang S / F |
|---|---|---|
| decode 1 | 89.62 / 98.75 | 170.35 / 181.62 |
| decode 16 | 164.17 / 259.79 | 263.20 / 445.23 |
| decode 32 | 195.24 / 370.01 | 253.78 / 621.18 |
| verify 1 | 135.59 / 152.38 | 261.50 / 277.86 |
| verify 16 | 296.80 / 423.24 | 520.02 / 714.14 |
| verify 32 | 392.58 / 627.99 | 740.72 / 1089.95 |

Per-stage **profile rows**, us (`2bbb626c`), copied without summing. Commas separate
rows; each time includes the parenthesized call count; shared dense names cannot be assigned reliably to roles.
Dense is projection work; MoE includes its route metadata/finalizer. PDL residence overlaps; omitted rows stay omitted. These are not critical-path times.

| Shape | Dense | Attention | MoE | Boundaries | Route | Other / omitted |
|---|---|---|---|---|---|---|
| decode 1 S | 17.6, 9.6, 7.5, 7.2, 4.7 | 10.9 | 26.6 | 5.0 (2) | 4.3 | — |
| decode 1 F | 17.9, 9.1, 7.6, 6.8, 4.1 | 20.7 | 33.9 | 5.2 (2) | 4.2 | — |
| decode 16 S | 47.3 (3), 10.2, 6.3 | 12.8, 1.4 | 73.9 | 15.3 (2) | 4.0 | — |
| decode 16 F | 53.1 (3), 41.7 | 105.4, 1.4 | 73.9 | 15.2 (2) | 4.1 | — |
| decode 32 S | 31.0 (2), 22.5, 6.6 | 17.5, 1.6 | 105.4 | 24.5 (2) | 4.1 | — |
| decode 32 F | 29.8, 27.2 (2), 22.5, 22.4 | 199.2, 1.6 | 92.2 | 31.0 (2) | 4.1 | — |
| verify 1 S | 32.9 (2), 10.4 (2) | 9.8, 4.3, 4.1, 2.2, 1.1 | 79.3 | 5.4 (2) | 3.9 | — |
| verify 1 F | 33.0 (2), 10.2 (2) | 19.0, 7.3, 3.3, 2.8, 1.2 | 88.0 | 8.4 (2) | 4.1 | — |
| verify 16 S | 26.1, 19.7, 14.4, 7.6 | 14.8 | 67.2, 54.8, 43.6, 39.4, 35.3, 33.6 | 12.2 (2) | — | 7 omitted rows |
| verify 16 F | 26.5, 19.7, 14.1 | 150.5, 22.4 | 72.8, 48.9, 46.5, 45.8, 42.5, 34.9 | 10.9 (2) | — | 4 omitted rows |
| verify 32 S | 34.3 (2), 32.8, 23.0 | 20.0 | 114.9, 81.2, 76.0, 51.9, 42.4, 32.3, 5.2 | 16.5 (2) | — | 6 omitted rows |
| verify 32 F | 35.0 (2), 32.7, 23.5 | 264.8, 18.9 | 96.2, 95.6, 74.1, 56.1, 31.9, 31.8 | 16.7 (2) | — | 3 omitted rows |

### Stage accounting and the lever model

`problems/muse_decode_layer/dev/levers/current.json` is explicitly S11 `acda6c62` / `863f502c`, not S14. Its caps and
per-role allocations cannot be carried forward as current measurements. A saving dS/dF per layer
changes score by `prod_load[T/(T - 46*dS - 16*dF)]^(1/6)`. Counterfactual levers do not add.
Profile sums include PDL waits; forked MoE overlap and omitted rows prevent exact attribution.
The boundaries' decode 16/32 residence excess 19.95 / 24.93 us ("+7.2%") is not critical path:
CUPTI increments were 5.15 / 5.92 us on different fixtures (`problems/muse_bf16_proj/dev/boundary_attribution/`).
The original-format ideal whole-layer ceiling is 3.344x (`sim/bound.py`); lossless coding
lowers those compulsory bytes. The measured stream roof is 2.36 us + bytes / 4.58 TB/s (`7b6c96b5`); ideal resource floors
are 4.8 TB/s, BF16 989 TFLOP/s, FP8 1,979 TFLOP/s. A stage cap near zero assumes fusion,
not a realizable isolated glue kernel. Whole-layer excess includes intrinsic work, not just launches.

Resource-floor table (us/layer, original uncoded dense bytes; model expert counts). Lossless
coding lowers the dense byte floors. Small attention/glue floors exclude their fixed work.

| Stage | M=1 | M=8 | M=16 | M=32 | M=128 | M=256 |
|---|---:|---:|---:|---:|---:|---:|
| QKV + gate | 7.86 | 7.86 | 7.86 | 7.86 | 7.86 | 9.77 |
| O | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 4.34 |
| Pool + router | 15.73 | 15.73 | 15.73 | 15.73 | 15.73 | 19.54 |
| Unpooler (T <= 8: the 1024-row slice) | 1.75 | 1.75 | 13.98 | 13.98 | 13.98 | 17.37 |
| Attention S / F | 0.11 / 4.33 | 0.11 / 4.33 | 1.77 / 69.3 | 3.55 / 138.7 | 1.78 / 69.3 | 3.56 / 138.7 |
| MoE | 14.7 | 44.2 | 43.9-44.9 | 52.7-53.4 | 108.4-109.1 | 125.7-126.1 |
| Boundaries, route | <= 0.02 | <= 0.02 | <= 0.02 | <= 0.02 | <= 0.02 | <= 0.02 |

## MoE M=1-32

Current nomination rule (97e98f3): every valid per-shape point estimate below1 is handed to P4,
with ratio,SE,seeds and A:A. Only P4's held-out six-load R lower confidence bound above1 promotes.
No per-load/row uncertainty veto or strict routing-identity check applies. `point_nominations.json`
and `review/point_*.patch` contain every earlier point winner and its exact patch; old confidence
calculations below describe uncertainty only.

### Common to every M

SGLang standalone (`ea3d5b6a`), M1/8/16/32/128/256: 77.3 / 142.5 / 192.0 / 207.2 /
457.3 / 620.7 us, 15-34% of floor. Routing model touched experts: standalone
2.65 / 8.46 / 11.9 / 15.9 / 22.7 / 24.4; layer 2.50 / 7.50 / 7.53 / 9.00 / 18.4 / 21.4
(`sim/routing.json`, `sim/problems.py`; layer counts inferred from official floors).
One read at 4.8 TB/s: M1 11.8-17.7 us (2-3 experts), M8/16/32 ~44 / 45 / 53 us.
Routes must match: unchanged NT1 verify 16 sliding 258.9 vs 304.6 us and verify 32 full
449.5 vs 546.6 across seeds (`0e94268c`, `7b2165c8`).

Lab fixtures need the layer's router bias/norm distribution (noise .045, mean alignment .07,
bias .055), not iid FP32 projection-host weights. Verify rows 1-7 share drafts per8-token block.
Only Marlin/Humming in the image accept BF16x NVFP4 on H200; MXFP4 group 32 E8M0 is not
losslessly interchangeable with group 16 E4M3. Library adapter failures are in "Failures".

### M=1-32

Current kernel: recoded one-LOP3 NT1, 8-token blocks, 256 threads, two 36 KiB stages,
grid 264, 125 registers, no spills; flags 95 at M1, 69 at M8/16/32.
Z zeroes norm sums in tile-0 before release; SLOT dispatches top-8 by popcount at M1;
ZOUT drops redundant post-norm loads; PDL overlaps router/prologue. M1 flags95 also include RSQ and NREG. Map's paired router+MoE
savings: M1 -5.58 +- 0.08 of 42.27 us, M8/16/32 -4.11 +- 0.13 / -4.83 +- 0.33 /
-4.46 +- 0.37 of 74.66 / 86.22 / 115.44 (`cf1de160`).

M1 timeline (`849e4017`, read flush, one seed): event 36.21 us = CUPTI node 31.81 +
4.40 framing; flags 95 node 28.14, CTA span 27.42, zero wait 0.03, route 1.50, barrier 0.74.
Pure stream same 56.6 MB node 16.69 us (3.39 TB/s); remaining 11.5 us: route 1.5,
first-tile issue/land 1.0/2.7, W2-ready 4.7, finalizer ~2.4. Zero-write-flush twin
`28fc0380` agrees within 0.5-1.5 us/phase.

M32 flags 69 critical CTA (`c768a8e3`): route 4.2, W13 36.1, W2-ready 11.6, W2 17.6,
grid barrier 5.0, combine 1.2 us of 76 us span. Wait ends ~4 us after last W13 end:
publish/fence/atomic under load plus synchronized expert completion cost ~12 us.
Extra 8-token blocks cost ~7.4 us each despite L2 weight rereads; capped-expert diagnostic
bounds reuse savings ~9 / ~35 us at M16/32 (`cf1de160`).

Measured roofs (`360810b1`, 16 seeds): M8/16/32 flags 69 64.9 / 75.9 / 104.7 us
(3.07 / 2.65 / 2.08 TB/s); trivial dequant ratios 0.930 / 0.913 / 0.828;
loads-only on same depth 2 pipeline 0.745 / 0.723 / 0.668; pure streams 49.4 / 50.8 / 53.2 us
( ~4.0-4.1 TB/s). Dequant 7-17%, math/sync/ready 10-16%, shallow pipeline 9-16 us short
of streaming. These diagnostics overlap, not additive isolated costs.
Reuse rebuild phase clocks at M32: math/activation 58.7 us, epilogues 28.5, exposed weight
waits 3.6; extra-block activation gathers replace first-block register reuse (`b5659458`).
Evidence: `problems/muse_decode_layer/dev/nt1_timeline/`.

### P3a reset receipts and decisions (2026-10-26)

Ratios are candidate time / control time, at M8 / 16 / 32 unless specified; below one wins.
All three official hosts pass, all requested seeds complete, and no lab is truncated. Correct arms
pass before/after native TASK gates (worst total/row <.00247 versus .015/.075 limits).
First/second/pipe S14 A:A mean absolute drift is .415/.235/.225, .442/.363/.322 and
.443/.250/.336 us respectively. First-run M1 A:A is .243 us; PAIR (.99890 SE.00232,n16) and NORM (.99839 SE.00213,n16) now nominate by point estimate.
These A:A summaries are not maximum per-seed envelopes. Exact rows, hashes and reproducible admission
checks: `dev/nt1_timeline/h200_results.json`, `read_reset.py`; complete pipe matrix: `h200_pipe.txt`.

| Recipe | Ratio versus control | Mechanism / decision | Leaves open | Receipt job |
|---|---|---|---|---|
| EPBASE / S14, 16 seeds | .99656 / .99979 / 1.00208 | Diagnostic hooks disabled; neutral control within uncertainty | Small phase differences are not speedups | `d650c04ead114119` |
| VEC / S14 | 1.01332 / 1.01457 / 1.01898 | Single vector writer loses despite fewer modeled bank requests; close this recipe | Other writer ownership and producer overlap | `d650c04ead114119` |
| PROD / S14 | .98056 / .97831 / .98517 | Move TMA issue to warp7; wins but PAIR is better | Whole-layer effect; no independent producer-only gain beyond PAIR established | `d650c04ead114119` |
| NORM / S14 | .98142 / .98340 / .99000 | Aggregate norm atomics per run; wins versus S14, adds no advantage over PAIR | Different norm ownership, not this tested combination | `d650c04ead114119` |
| PAIR / S14 | **.97493 / .97262 / .97932** | Two float2 planes/two writers plus warp7 producer/per-run norms; 16/16 wins each M; nominate to P4 | Many-seed six-load whole-layer pair; M1 unchanged (.99890, neutral) | `d650c04ead114119` |
| ONCE / S14 | 1.30467 / 1.29879 / 1.33132 | Decode-once schedule remains a loss | Other reuse schedules; the 58.7 us old math clock is ONCE, not S14 | `d650c04ead114119` |
| ONCE_VEC / S14 | 1.23515 / 1.24645 / 1.31428 | Vector epilogue improves ONCE, still loses to S14; close this combination | Reuse without extra activation-gather cost | `d650c04ead114119` |
| ONCE_PAIR / S14 | 1.24614 / 1.25036 / 1.29410 | Pair epilogue does not rescue this reuse schedule | Alternative reuse mechanisms | `d650c04ead114119` |
| PAIR / S14, 32-seed confirmation | **.97830 / .97462 / .97778** | 31/31/32 wins; confirms first run beyond A:A and upper95% summary bound | P4 integration and whole-layer interaction | `29ad388de976432c` |
| TREE / PAIR (and / S14) | .99904 / .99811 / 1.00007 (.97888 / .97404 / .97746) | Balanced FADD depth8→3 has point wins over PAIR at M8/M16; hand those pieces to P4 | Held-out whole-layer effect of the small point gains and other dependency cuts | `29ad388de976432c` |
| FOUR / PAIR (and / S14) | 1.02215 / 1.02842 / 1.03432 (1.00178 / 1.00521 / 1.01047) | Four half-active writers spend more instructions; close exact recipe | Other bank/ownership layouts | `29ad388de976432c` |
| LOCAL / PAIR (and / S14) | 1.01213 / 1.00951 / 1.00939 (.99212 / .98615 / .98488) | Removing global norm atomics costs export/final gather; worse than PAIR but a point winner versus S14 at all3 M | Different reduction with less export/tail cost | `29ad388de976432c` |
| FOUR_LOCAL / PAIR (and / S14) | 1.03406 / 1.04155 / 1.04617 (1.01140 / 1.01404 / 1.02290) | Combined four-writer/local-norm ownership loses; close combination | Neither general atomics removal nor all ownership rebuilds are closed | `29ad388de976432c` |
| PDIA / S14, deliberately wrong math | .90404 / .89884 / .82615 | Removing decoder/MMA while retaining loads exposes headroom; **not a correctness-gated candidate** | No promotable gain; retained sync/epi still costs time | `16c180784f414576` |
| Hot-L2 / cold-HBM math probe | .87069 / .94082 / .96439 | Cold math is slower; not sufficient to identify inadequate flight | Cache working set and interaction with full-kernel scheduling | `16c180784f414576` |
| COPY depth4 / depth2, requested cap2 | .98253 / 1.07285 / 1.14805 | Deeper copy has no consistent gain and changes actual residency (depth4 fits1 CTA/SM) | Full-kernel lookahead; no FLIGHT nomination from this contrast | `16c180784f414576` |
| Shared MMA ILP2 / W8, same2 CTAs/16 warps | .98991 / .98145 / .98272 node time; work-normalized .98973 / .98177 / .98275 | Small dependency contrast supports sending prepared packed rebuild lab; no saturated pipe established | Microprobes lack independent A:A; full-kernel gain unmeasured | `16c180784f414576` |
| Shared W16 / W8, same2 CTAs, work-normalized | 1.01796 / 1.02019 / 1.01646 | 32 active warps do not improve normalized work over16; no K16 nomination | Full K16 kernel differs from microprobe and remains unmeasured | `16c180784f414576` |
| TMA36KiB hot / cold completion, grid1/132/264 | .47337 / .51825 / .52800; cold .676 / 1.096 / 2.000 us | At264 CTAs, 72KiB lookahead / 60,606 required unique-stream bytes =1.217 coverage | All-miss repeated-load traffic and average outstanding bytes unmeasured | `16c180784f414576` |

PAIR's summary confidence calculation clears1 at every affected M, but that is descriptive, not a
nomination requirement. P4 now receives all point wins: ILP at M8/16/32; TREE at M8/16 versus PAIR and
M8/16/32 versus S14; LOCAL at M8/16/32 versus S14; PROD/NORM alternatives and M1 PAIR/NORM.
`point_nominations.json` includes ratio,SE,seeds,shape policy and exact patches; P4 composes and
confirms by R. EPBASE's point-winning control aliases are included for deduplication. Phase probes have 4–8% observer tax and overlapping intervals;
their subclocks cannot be added as isolated costs. SHF/PRMT cap8 dependent-probe clocks are anomalous
(1.336/1.447 GHz versus ~1.96 elsewhere), so their >2 issue rates are not used as pipeline limits.

Response sent immediately after classification: unchanged `prof/lab/moe_nt1_pipe_rebuild.py` `caa59593`,
all four FLIGHT/ILP/K16/LUT arms,32 seeds/M, through `h200_lab.py` on MoE. Sender **4124752**, merged
wrapper **`de47736b9f588673`** with waiting P3b round2 payload; `dev/nt1_timeline/followup_queue.json`
tracks dispatch. The small ILP contrast motivated this full paired test; no individual pipe is proven
to be the full-kernel bottleneck. Its measured outcome follows; PAIR's whole-layer effect remains unverified.

### P3a four-arm rebuild receipt (2026-10-26)

`dd539517bc8344dd`, wrapper `de47736b9f588673`, payload `caa59593`: official host passes all5
correctness tests, with no flags. All32 seeds complete at M8/16/32, every arm's native TASK gates pass
before/after (total .00238, worst row .00246), and the full merged output is retained. S14 A:A mean
absolute drift is **.487/.316/.314 us**, about .667/.398/.282% of the ILP paired controls.
Ratios below are candidate/S14 time; lower wins. Exact hash/control audit: `read_rebuild.py` and
`dev/nt1_timeline/rebuild_results.json`.

| Recipe | Ratio versus S14 at M8 / 16 / 32 | Mechanism / decision | Leaves open | Receipt job |
|---|---|---|---|---|
| FLIGHT, bulk L2 prefetch beyond2 raw slots | 1.07662 / 1.05285 / 1.03119; 0/32 wins each M | +5.617/+4.229/+3.502 us. Extra prefetch does not rescue the pipeline; close this exact schedule | Other lookahead distances, issuing agents and schedules that avoid competing with useful TMA work; no hardware counter identifies the regression | `dd539517bc8344dd` |
| ILP, alternate2 MMA accumulator chains | **.99055 / .99041 / .99046**;26/30/30 wins of32 | -.680/-.765/-1.040 us. Upper95%+A:A =1.000762/.996981/.996081: nominate **M8/M16/M32** to P4; the old M8 A:A veto is removed | Held-out six-load effect, other chain/register schedules and combination with PAIR remain open | `dd539517bc8344dd` |
| K16/r64,16 warps/CTA and shared B cache | 1.20394 / 1.21153 / 1.23270;0/32 wins each M | +14.984/+16.942/+26.007 us despite measured2 CTAs/SM,64regs,16-byte local stack. Close exact partition/cache recipe | Other K partitions and occupancy layouts without this B-cache/epilogue cost; higher occupancy generally is not closed | `dd539517bc8344dd` |
| LUT, fixed1KiB FP4-pair lookup, scale math unchanged | 1.59959 / 1.62315 / 1.67379;0/32 wins each M | +44.005/+49.545/+75.436 us,2 CTAs/SM,128regs. Close this exact code-pair table/layout; no measured saturated-port attribution | Other lossless code layouts and decoder instruction reductions without this table/packing/shared-service cost | `dd539517bc8344dd` |

ILP's consumer nodes are67.083/72.868/104.667 us versus S14's67.652/73.791/105.861, consistent
with a small kernel gain. Neither the static128-register ILP footprint nor the diagnostic microprobe
identifies a unique physical bottleneck. The failed recipes do not close flight, occupancy or lookup
methods in general. All4 arms are implemented and gated, so their losses are measured; the old M8 ILP
margin calculation no longer vetoes its point nomination.

P4 handoff: `dev/nt1_timeline/pipe_s14_ilp.patch`, SHA256 `b55c2970a7737c3708b0e5f251ab24b31ef1888bdae55f812c8e155d9eac230d`,
onto exact S14 `f1c77ce6`; screened candidate `f71a0206`, cell `809bef8e`. The patch preserves M1;
P4 receives the **point nomination at M8/M16/M32**. PAIR is faster in the direct comparison below;
ILP remains a valid alternative to S14. `p4_ilp_handoff.json` pins full hashes, flags,4 affected sliding/full shapes and local
normal/forced admission. No whole-layer gain or additive PAIR+ILP gain is claimed.

Next sent: `prof/lab/moe_nt1_ilp_vs_pair.py` `d1e40f35`, sender**479906**, wrapper**`af7b4526022c8d75`**,
on MoE at priority1.2. Direct ILP/PAIR comparison plus S14 back,32 fresh seeds/M,readflush ABBAx6,
before/after TASK gates,A:A mean/max,CUPTI nodes. The exact payload passes all32 local seeds on GPU0,
with unchanged admitted native cells; `ilp_vs_pair_validation.json` pins its1,725-character output.
Its measured receipt follows; every whole-layer/composed gain remains unmeasured.


### P3a direct PAIR confirmation

`1b135b70835f4634`,32 fresh seeds: official/native TASK gates pass,all seeds and output complete.
PAIR A:A mean absolute .488/.377/.364us,max1.424/.928/1.648 at M8/16/32.

| Recipe | Ratio at M8 / 16 / 32 (SE) | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| ILP / PAIR | 1.01094/1.02063/1.01190 (.00178/.00242/.00220),n32 | PAIR is faster directly. ILP remains a point alternative versus S14,including M8 | ILP+PAIR composition and held-out whole-layer effect; no additive claim | `1b135b70835f4634` |
| S14 / PAIR | 1.02243/1.02739/1.02071 (.00226/.00254/.00283),n32 | Confirms PAIR's gain on new weights/routes | Whole-layer R and interactions with the new review arms | `1b135b70835f4634` |

### P3a S15 dependency/identity/packet review

`review/claims.json` pins the actual S15 source60254883 and CUDAc023c946. S15's small-M CUDA is
byte-identical to S14's. Each accepted claim is one arm; shards have the already-measured PAIR parent,
which is also included as a control. New H200 performance is unmeasured until the packed batch returns.

| Recipe | Ratio / measured prerequisite | Mechanism / decision | Leaves open | Receipt / evidence |
|---|---|---|---|---|
| BLOCK | H200 unmeasured; expert-wide `NT1*nblk[e]` wait confirmed in S15 | W2 acquires `done[e,nb]` againstNT1;128 done slots/512B at M32. W13 store→fence→barrier→publish→acquire preserved. Only M16/M32 | Actual wait reduction and counter-address cost on H200 | `review/claims.json`, `review/manifest.json` |
| CANCEL | H200 unmeasured; W2 `v*=256*gs2[e]` before post-norm/RMS confirmed | Drop that element multiply,adjust eps/a²,retain sign,guard zero scale/norm and handle epsilon range in double. W13/SwiGLU unchanged | H200 savings versus finalizer/metadata cost and six-load score | `review/extreme_validation.json`:160 helper cases; `component_validation.json`:94 TASK checks |
| SHARD2 | H200 unmeasured; PAIR's single norm slot confirmed |2 atomic slots per pair,CTA-indexed; W13 zeros both,finalizer sums2 | Contention relief versus extra zero/load work; no LOCAL export | `review/component_validation.json`,exact patch |
| SHARD4 | H200 unmeasured; same PAIR parent |4 independent norm slots per pair | Same trade-off at4 shards; interactions with block readiness | `review/component_validation.json`,exact patch |
| SHARD8 | H200 unmeasured; same PAIR parent |8 independent norm slots per pair | Same trade-off at8 shards; other reduction ownership | `review/component_validation.json`,exact patch |
| TWO_BASE | Same-work sm90a decoder integer instructions37→406,registers30→31;8,128 signed-format half checks pass | Identity confirmed,but this sign/exponent assembly grows integer issue;**dropped before H200** | More efficient packed exponent assembly; this does not close every two-base decoder | `review/decode_issue.json`,`audit_decode.py` |
| PACK5 | Actual official seed5353 weight tensors:100,663,296 scale bytes,all256-byte packets fit32 symbols; final byte cut**4.05816%** | Exact5-bit packets with per-packet raw fallback,32B tile header,8B offsets,timed TMA/decode. Full bank bit proof and raw transfers pass | H200 benefit after extra decode/metadata issue; other packet layouts and checkpoint distributions | `review/scale_compressibility.json`,`component_validation.json`,`raw_moe_validation.json` |

All7 native flags69 images retain32 HMMA/128 HMUL2,zero spills;125regs except PACK5127regs.
Components:90 TASK checks per arm,94 CANCEL,worst total/row.002391/.002481; raw-packet whole-MoE
adds9 captured checks. Local full payload:16 fresh seeds at each M1/8/16/32,64 independent A:A pairs,
TASK before/after,all7 arms and direct shard/PAIR comparisons. Complete seed-level archive uses
6,399 characters and took52.4s locally (`review/lab_local_validation.json`); local timings are not H200
claims. All14 normal/forced whole-layer screens pass13 cases,12 shapes and26 static checks each
(`review/screens.json`).
Failed OOM/proof attempts stay local and are not admission. `review/queue.json` records the screened
send: MoE sender**1191843**,wrapper**`a2344c10505b0e44`**,priority1.2,import budget240s,
full9000-character allocation; prefill is excluded. All new H200 ratios remain unmeasured.

## MoE M=128

Current N_full family: NT1 <=8-token experts, N16G4 at 9-16 on a third stream, heavy N32
above 16, joined before distributed F4 finalizer. Light experts decode once; PERMW retained.
Initial N_full vs sequential H_H:207.40 vs 234.78 us, 0.8842 (`c01b1916`, 36/36 bit-equal).
Heavy first on side stream vs NT1-first:203.43 vs 207.60 us, 0.9799 (`880c3881`).
Heavy lean loop + interleaved gate/up rows (`858ff2fa`, `701d2867`): 0.9613 then 0.9923;
static KT128 stage 254→207 SASS, core 128, remainder ~79 descriptors/pointers/copies/LDS/sync.
Light N16G4 vs S12 final: 0.9328 at 4 CTAs/SM,196.2→182.4 us (`eea9d87b`);
S14 absolute control 185.4 us (`8ee003db`).

## MoE M=256

Same three-way policy. Original N_full vs mixed_parallel 298.68 vs 342.80 us, 0.8701
(`14f942d5`); heavy-first 295.71 vs 299.49, 0.9882 (`880c3881`). Lean loop 0.9426,
279.38 vs 296.72 (`858ff2fa`); gate/up interleave 0.9838 (`701d2867`).
Light N16G4 at 4 CTAs/SM 0.9485,270.6→256.4 us (`eea9d87b`);
S14 control 261.0 us (`8ee003db`). F4 isolated finish 12.8→9.1 us, 0.9850 /0.9885
whole MoE at M128/256 (`5e197b5e`).

What binds the family (`3e8996f8`; S12 branches alone, us kernel/trivial-dequant/loads/stream):

| M | NT1 W13 | NT1 W2 | Heavy W13 | Heavy W2 |
|---|---|---|---|---|
|128|74.9 /69.9 /65.0 /61.4|44.5 /40.7 /36.0 /33.9|56.4 /50.6 /38.1 /29.2|37.1 /32.9 /25.1 /17.4|
| 256|78.4 →64.4 stream|46.2 →35.4 stream|109.5 /94.4 /70.2 /50.3|60.7 /54.6 /39.5 /26.8|

NT1 1.22-1.31x stream, heavy 1.9-2.3x; heavy schedules 1.6x /2.2x unique weight bytes
and rereads 171 /403 MB activations. Unique MoE at M128 ~467 MB ( ~102 us at 4.58 TB/s)
vs ~171 us in-layer. These diagnose the predecessor; light N16 changes its mix.
Branch overlap was small (`5e197b5e`): heavy 99.8 /177.7 us, NT1 127.6 /133.7,
together 195.0 /275.0 at M128/256. Inferred: heavy fills SM shared memory ( ~57 KB x 3),
NT1 ~87 KB CTAs wait; co-residency restrictions hurt bandwidth. Static issue roof 4.21 TB/s
assumed 128 lane-ops/cycle (`8483c0c6` audit), distinct from the hardware 64 integer throughput.
Live family: `problems/muse_moe_w4a16/dev/mixed_wgmma/branches/`.

Reset heavy-branch experiments (ratio candidate/S14, M128 / M256; <1 wins):

| Recipe | Paired ratio | Mechanism / evidence | Leaves open | Receipt |
|---|---|---|---|---|
| Decode-once DIRECT, cluster1 launch |1.2039 /1.1427|16 seeds x3; all outputs bit-equal; active cluster counts396/264 match3/2 CTAs perSM, so cluster1 does not impose one CTA/SM|Cheaper issue/transport at N32; another reuse ownership|`8792099636654bac`|
| Same DIRECT image, ordinary CTA launch |1.1965 /1.1381|Nearly the same loss as cluster1; fixed heavy phase106.0/182.9us vs S14's78.5/128.3us; topology is not the main penalty|Weight/decode reuse with lower per-stage issue cost|`8792099636654bac`|
| Same DIRECT image, serial width classes |1.2585 /1.2992|Serial classes worsen whole-MoE timing; isolated heavy125.95/194.75us|Parallel classes with cheaper kernels|`8792099636654bac`|
| DIRECT serial, S3 for MAX64 |1.1976 /1.2273|Improves serial by~4.9/5.5%, still loses to S14; does not establish S3 on parallel CTA2|Queued CTA2S3 direct pair; another pipeline/ownership|`8792099636654bac`|

`8792099636654bac` controls: S14 182.90/256.67us, A:A signed/absolute/max difference
+0.043/0.519/3.120us across both M, worst FP32 L2/row0.004134/0.004709. All arms48/48
bit-equal at each M. Ordinary DIRECT's ratio1.1965/1.1381 agrees with the prior1.1970/1.1322
calibration (`8ee003db`) within0.6%; isolated heavy-phase totals also remain within~2%. The measured
loss closes these topology changes, not decode-once itself. Do not allocate the old residual
into causal decode/epilogue/overlap components from isolated phases and a different mean input.

## Attention

### Shapes, floors and the incumbents

16 Q heads, 2 KV heads, d64, FP8 KV, page1, ctx 80K (agentic +-7.5% at B>1).
Full B16/32 reads 327.68 /655.36 MB, 69.3 /138.7 us at 4.8 TB/s; official 63.4 /126.8 us
use shortest context. q8 same KV bytes, 41.95 /83.89 GFLOP extra, 21.2 /42.4 us at FP8 peak.
SGLang FA3 (`cbebc3d4`): sliding 20.5-34.3 us, 1-11% of floor; full 31.6-391.5 us,
fit 21.8 us + bytes/1.66 TB/s. q8 full near q1 time: KV path, not extra math, binds FA3.
Image FA3 takes one dtype for Q/K/V: FP8 cache requires E4M3 Q.

Current: q1 `attn_main` gather; decode 1 reads projection partials directly; q8 FA3 public
`layer` scheduler except vq8 page ring at full verify 16/32. Direct-partials full-layer
ratios 0.971 /0.9997 /0.9990 at decode 1/16/32 (`c114c35e`): retained only decode 1.
Attention stage control (`8fd0d2a6`, predecessor, projection stub subtracted), us S/F:
decode 1 11.6/19.5, decode 16 8.3/99.9, decode 32 12.6/190.1;
verify 1 16.5/29.5, verify 16 15.9/142.7, verify 32 21.7/253.1.

Sliding's decode 1 chain is dependent latency, not just launches (`79061182`, `c0fd644c`):
entry/QKV wait 3.6, prep 1.2, history gather 5.3, QK 1.8, softmax 0.9, PV 0.2, combine/epilogue 1.8
of 14.4 us. History does not depend on current QKV. Full verify vq8 spends ~907 of ~1,690
cycles/64-key tile in softmax (`f4d1a195`).

S15 stage-only winners: last-CTA N128 sliding decode 1 0.980 vs direct-partials
(-1.43 us, SE 0.0013; `7e93cc81`); register-fold N128 full verify stage 4
0.977 /0.948, critical 152.9→141.4 /265.4→229.7 us (`9d9a01e1`).
Live designs: `problems/muse_attention_decode/dev/cluster_sliding/`,
`problems/muse_attention_decode/dev/early_history/`, `problems/muse_attention_decode/dev/full128/`,
`problems/muse_attention_decode/dev/full_attention/`.

### P5 sliding and decode-1 receipt decisions

Ratios are candidate/control time. Stage-only wins require a paired whole-layer
confirmation; a missing correctness/control gate is unmeasured, not a loss.

| Recipe | Ratio vs control / A:A | Mechanism established | Leaves open (and what it closes) | Receipt |
|---|---|---|---|---|
| Decode-1 full, one-head N128 persistent gather (stages 3/4; compact eight-row partials) | H3/H4/GC 1.4240/1.3879/1.5046; A:A 0.99029 (SE 0.00174), eight seeds; fresh BF16/partial errors 0.0047-0.0059 | Per tile: producer issue 1714, softmax 907, publication 695 cycles in the instrumented H3 shader | Closes these stage schedules, not one-head full attention; no P4 promotion | `58d6a59603564014` |
| Sliding LAST/MASK/FAST, six shapes, eight seeds | LAST d1/16/32 0.9307/1.0523/1.0718, v1/16/32 0.9030/0.8618/0.7997; A:A 0.9856/0.9890/0.9884/0.9968/0.9923/0.9894. FAST verify 1.0503/1.4051/1.2840; all eight stage/strided-partial gates per shape pass | Padded masks are bitwise equal; FAST changes RoPE math and does not improve verify. LAST removes a launch but still serializes final reduction | Closes these FAST verify and LAST d16/d32 stage recipes; query partition, DSM and resident-history rebuilds remain open. Whole-layer v1 decision is in round2, not a P4 promotion | `2f8b702dc76b42fd` |
| Global shared-merge last-CTA, frozen S11, sixteen seeds | **Unmeasured whole-layer recipe**. Raw C128 d1S/d16S/d32S/d1F 0.98987/0.99595/1.00529/1.07796; A:A 1.00514/0.98012/1.00524/1.01041 (SE 0.00475/0.00359/0.00210/0.00298) | Only the isolated full-QKV first fixture has an attention/store gate; captured arms check finiteness, not TASK. Sliding output drift 0.15-0.58 and routing changes require full validation | Closes no correctness-proven recipe and promotes nothing. Round2 tests the sliding shapes with full TASK gates; retain raw global timing only as diagnostic evidence | `ae6cd27877cc4ee2` |
| Native sliding verify DSM C1/2/4/8/16, eight seeds | Best v1 C16 0.71927 (vs A2 0.74918), v16 C8 0.84911 (0.86657), v32 C4 1.07381 (1.08299). A:A 0.96012/0.97988/0.99157; SE 0.00336/0.00328/0.00309; ranges 0.94072-0.97302 / 0.96656-0.99095 / 0.97661-1.00294. All eight stage gates pass; max error 0.02213/0.02121/0.02066, strided partial error 0 | Distributing output rows over native cluster CTAs removes last-CTA serialization; the winning gang is C8 at v16, not the original one-wave C4 proposal. No isolated phase attribution in this run | Send selected B1:C16 and B16:C8 RESET/COOP/half/double whole-layer confirmation, 24 held-out seeds. Close only this v32 DSM sweep; resident-history/QT rebuilds remain open. No P4 promotion from a stage-only win | `b05713ecf6574ad9` |
| Compact early sliding history, exact S14, E4/E2/E1 and late L4 vs exact OLD4; 16 seeds x12 rounds | E4/E2/E1/L4 0.997414/1.016159/1.040193/1.020247 vs OLD4; best E4 SE 0.001244. A:A 1.009555, SE 0.006105, seed range 0.98824-1.09546 (noisy). All seven arms pass all16 TASK gates with20% margin and replay checks | Prewait work is implemented: QT4 transpose median/max 4.064/4.608us precedes wait completion 5.152us; shared 53504→28928B. E4 attention increment 12.160 vs OLD4 11.200us, QKV/gate 7.296/1.536 vs8.064/2.464us; no isolated cause of the neutral total | E4 gain is below A:A drift and3SE: drop the compact last-CTA history recipe, send no history confirmation, promote nothing to P4. Resident verify and cooperative/QT rebuilds remain open | `402819c28279466d` |
| Selected DSM whole-layer confirmation, exact S14; 24 held-out seeds x12 rounds | **v1 unmeasured: identical-recipe control fails.** Nominal RESET/COOP/HALF/DOUBLE 0.96309/0.94983/0.95830/0.98687; A:A1.00963 (SE0.00201), but DOUBLE/COOP1.03901 (SE0.00300) for the same C16 shader/arguments. v16 RESET1.00412 (SE0.00395), COOP/HALF/DOUBLE1.11715/1.06338/1.15331; A:A0.999657 (SE0.000744), range0.99406-1.00701. All six arms x24 TASK/cache gates pass | v16 cooperative QT1 uses254 registers/2 active CTAs vs RESET168/3; attention increment51.648 vs24.128us baseline and25.696us RESET, consistent with register pressure, not an isolated attribution. v1's fixed-capture discrepancy is a real failed control, not proof of a kernel gain | Close only these v16 cooperative QT1/gang recipes; old RESET is neutral. Promote no v1 arm yet. Repeat v1 with35 disjoint seeds, A/A2/A3, identical C16 arms, fresh graph pools and rotated capture ownership; resident bounded-prep verify remains independent | `6c26b14fdd28458b` |
| Verify1 DSM control repeat, exact S14,35 disjoint seeds x14 rounds; graph pools recaptured with rotated ownership each seed | COOP/duplicate0.951214/0.954611; pooled identical C16 **0.952913**, SE0.005071. A2/A3 0.999262/0.999499 (SE0.004232/0.003422); A2 range0.93838-1.05208. Duplicate/COOP1.003740, SE0.004151: control passes. All7 arms x35 TASK/cache gates pass | First-capture attention increment17.632→12.064us, QKV delta-0.032/O+0.352us; both identical captures qualify against both baselines. Fresh-capture cohort resolves the prior control discrepancy; its cause is not isolated from the changed cohort | **Promote verify1 sliding cooperative DSM QT4/C16 to P4**, shader0f250847. S14 integration candidate403bb778 passes normal/forced13 TASK/12 graph/24 static each (local DSM C1 stand-in). `dev/sliding_round2/handoff.json` rebuilds it; P4 current-base composition and six-load score remain unmeasured | `6a0b3f0987e94ba5` |
| Round2 last-CTA/query-partition/early/DSM matrix; exact S14,16 seeds x12 rounds | Best d16 D8 0.995283, d32 D4 0.989821: neither qualifies; A:A1.007797/1.002142 (SE0.001748/0.000707). v1 D16 0.924229, A:A1.009469 (SE0.002417); all shape-arms pass all16 TASK gates. Raw v1 L8Q8/L16Q8/DQ8/RESET_D16 0.95003/0.96214/0.95044/0.93006 | v1 last-CTA QT1 attention increment76.608us vs QT8/split8 12.864us and D16 11.840us; prep/epilogue row ownership matters. D16's shader and S14 base hash-match the35-seed control repeat | No d16/d32 promotion; close only these losing tilings, leave other one-launch schedules open. Best v1 D16 is already qualified by6a0b3f09: use that P4 handoff without a redundant selected-QT run. Other v1 arms remain unpromoted | `27a914f72bfb47c8` |
| Resident-history verify, C2/4/8/16 pipeline, C4/8 serial and RAW4/8 edge controls;24 seeds x12 rounds | Best v16 C8S1.012806, v32 C4S1.020146; A:A0.999296/0.998184 (SE0.000342/0.000252; ranges0.99586-1.00187/0.99557-0.99991). All10 arms x24 TASK/cache gates per shape pass; no qualifier | Native edge advances v32 C4P start to2.847us vs QKV end20.063us (RAW4 starts20.128us). QKV uses221508B shared, leaving11964B of233472, less than any tested attention CTA (min27648B): broad co-residency is impossible on this producer. v16 C16 starts4.513us before QKV end13.792us but still loses1.05542. Per-CTA preload overlap is not measured | Drop these resident/serial/edge recipes, send no confirmation, promote nothing. This does not close resident attention with a lower-footprint producer, different gather/transpose or output schedule | `857f78fa889f4646` |

### Our kernels (losses in "Closed recipes")

vq8 uses one persistent 384-thread CTA/SM, stream-K over request/head-pair/tile;
producer prefetches page IDs 4 tiles ahead, cp.async + V transpose; consumers native FP8 WGMMA
QK SS/PV RS (P in registers), QK(j+1) before PV(j), skip rescale if max unchanged;
PDL combine RMSNorm/sigmoid. 168 registers, no spills. Publish-first ring LAG=ST-2;
ST-1 deadlocks. Page ring `c262e4c5` vs FA3 (`5928ced8`): full verify 16 141.6 vs 191.9 us,
verify 32 250.6 vs 378.9; not q1 (132.2 /239.6 vs ~104 / ~191).
Original serialized ring period 2677 /2883 cycles, consumers wait 51-55%, producer 46-48%;
publish fix reduced B32 ~1800 (model floor 1400; `44aa67ba`).
FP8 `mma.sync` is emulated on sm90 (12 converts +2 FP16 HMMA under CUDA13.0.88);
native WGMMA is required for FP8 peak (`be19da94`, static). Packed-half exp lowers to 2 MUFU,
so saves no issue. Native Q-major SS design spent ~990 cycles V transpose/barrier and ~760
softmax vs QK ~200/PV ~208: issue, not memory, made it lose (`a1679f9c`).


### Full-attention reset measurements

`457f3c73b3214534`: native full verify16/32, S14 N64 control,3 fresh seeds ×16 cold rotating
replays; all probability/cache/TASK gates and native tensor-sum probes pass. A:A means0.99941/0.99947,
seed ranges0.99562–1.00164 /0.99608–1.00154. Ratios are candidate/control time; no arm is promoted.

| Recipe | Ratio verify16 /32 | Mechanism / evidence | Leaves open | Receipt |
|---|---:|---|---|---|
|25% FMA-polynomial /MUFU split|1.05012 /1.04610|Exp/sum357.8→386.3 /360.4→387.5 cycles; extra issue outweighs SFU relief|Different software-exp implementation or issue schedule; this compiled split loses|`457f3c73b3214534`|
|50% FMA-polynomial /MUFU split|1.08436 /1.07352|Exp/sum476.6/473.4 cycles, above native357.8/360.4|Other approximation/reduction implementations; not all mixed-pipe designs|`457f3c73b3214534`|
|One800-cycle initial consumer-WG stagger|1.00004 /1.00126|Inside A:A spread; existing rescale skip taken70.3/80.5%|Sustained overlap without a fixed initial stagger|`457f3c73b3214534`|
|25% split +initial stagger|1.05200 /1.04865|Stagger does not hide polynomial issue cost|Other overlap structures|`457f3c73b3214534`|
|FP8 P×ones tensor-core row sums +25% split|1.03473 /1.03452|Alpha/sum157.4→72.2 /155.7→72.9 cycles; separate WGMMA group/wait remains; pure TC effect not isolated|MUFU-only TC sums, denominator accumulated inside the existing PV group; N128 incumbent|`457f3c73b3214534`|
|Explicit two-consumer-WG ping-pong|1.15929 /1.17091|Clocked QK wait75.2/75.7 vs13.5/14.4; added synchronization|Other intra-WG overlap schedules; this turn protocol loses|`457f3c73b3214534`|
|Ping-pong +25% split|1.22654 /1.21961|Both added synchronization and polynomial issue|Independent scheduling/exp implementations|`457f3c73b3214534`|


`e4fcf732f35b425d`: native full decode16/32,8 fresh seeds ×8 cold rotating replays;
all independent probes and TASK/cache gates pass. A:A means0.99964/1.00105,
seed ranges0.99310–1.00543 /0.99778–1.00871. No arm beats A:A; no P4 promotion.
Clocked gather/page issue1539/1597 cycles dominates softmax618/615 (instrumentation perturbs wall time).

| Recipe | Ratio decode16 /32 | Mechanism / evidence | Leaves open | Receipt |
|---|---:|---|---|---|
|attn_main25% FMA/MUFU split|1.00182 /1.00284|Inside A:A spread; exp/sum103→92 /102→91 cycles saves too little|Gather/issue and other software-exp schedules|`e4fcf732f35b425d`|
|attn_main50% FMA/MUFU split|1.00898 /1.01133|Exp/sum falls but page/issue and total time rise|Different issue/transport structures|`e4fcf732f35b425d`|
|FP16-P tensor-core row sums|1.00976 /1.00704|Exact half-P denominator probe passes; packing/sum29→95 /29→94 cycles|Denominator fused with existing PV work, no separate MMA/wait|`e4fcf732f35b425d`|
|TC sums +25% split|1.00814 /1.00896|Packing overhead remains, no overall benefit|Other denominator schedules|`e4fcf732f35b425d`|
|Native two-WG q_len1 control|1.33968 /1.28368|Padded verify-query work at q_len1; exceeds four-warp incumbent|A q_len1-specialized WG design, not this padded transfer|`e4fcf732f35b425d`|
|Two-WG25% /50% split|1.39793/1.33255;1.44732/1.37264|Adds polynomial issue on an already slower family|Other WG/Q layouts and exp implementations|`e4fcf732f35b425d`|
|Two-WG ping /ping25|1.68078/1.63104;1.74657/1.67450|QK wait78/79 cycles plus the padded-query cost|Other overlap protocols|`e4fcf732f35b425d`|


### P5b S15 audit reset receipts

Ratios below are new/old time. Six-load order is d1/d16/d32/v1/v16/v32.
The frozen audit payloads retained per-load mean/SE/ranges, not joint seed times;
there is no exact six-load confidence interval to promote from these summaries.
Treat d1 as unmeasured for both audit recipes: every A:A seed ratio exceeds1,
so its drift cannot establish a d1 gain or loss.
Held-out confirmation retains weighted per-load times for every seed. PDL's d1
A:A is biased1.009834 (range1.000283–1.016426); other five A:A means are
.997998/.998476/.997318/.998725/.999697, ranges .994593–1.001740 /
.996282–1.000505 / .993044–1.000308 / .994851–1.002511 / .997835–1.002037.
Library's d1 A:A is1.013075 (1.004542–1.022814); other means are
.998606/.999222/.998548/.999198/.999509, ranges .992040–1.008489 /
.996336–1.002392 / .993566–1.003576 / .997946–1.000551 / .997735–1.001910.

| Recipe | Ratio vs control | Mechanism checked | Leaves open | Receipt |
|---|---|---|---|---|
| S15e all release edits | 1.018308/.988901/.991881/1.001564/.977655/.992563 vs S15; E-identical no-glue1.018116/.987360/.991442/1.002360/.979613/.993467 | Native changed sources execute; equivalence/routing gates pass all12 shapes×8seeds. Most old producers already released at entry | No wholesale promotion: d1 control drifts and joint seed confidence was omitted. Selected producers go to held-out TASK/R confirmation | `f88fda4aac144374` |
| q1 attention PDL consumer enable+wait | E/no-attention .992546/.993376 at d16/d32; A:A .997998/.998476 | Existing sumcast entry release can overlap the now-enabled attn_main prologue; wait precedes qkv reads | Nominate exact attention patch for confirmation; no change to decode1/sliding kernel math | `f88fda4aac144374` |
| Large-M MoE route/GEMM/phase/finish entry release | E/no-MoE .988392/.993228 at v16/v32 | PDL enabled with consumer waits and existing full-completion side-stream joins; native output/routing parity passes | Isolate large-M from small-M release in held-out confirmation | `f88fda4aac144374` |
| VDSK qkv producer entry release | E/no-projections .988099 at v16 | Original independent X/W prologue, downstream VQ8 wait retained; arithmetic unchanged | Confirm separately and with large-M MoE; unchanged-load variation forbids attributing all E gain to one producer | `f88fda4aac144374` |
| Small-M/generic V9 entry release in grouped E | E/no-MoE 1.007300/.999919/.998656/1.001767 at d1/d16/d32/v1 | Changed native code compiled/executed; outputs agree | Drop this tested small-M release from selected confirmation; different wait/prologue schedules remain open | `f88fda4aac144374` |
| S15L all library replacements | .999802/.999176/1.001312/1.006730/.994152/1.005131 vs S15; all768 TASK gates pass | Native NF O, VDSK pool/unpool and VQ8 rows replace external completion boundaries | Full composition is not a winner; isolate candidates below, do not blame all libraries or all native kernels | `f22020d25d234be6` |
| Owned NF O128/O256 | L/no-O .995180/.998636 at v16/v32; A:A .999198/.999509 | Same BF16 inputs, owned release/wait chain; exact native kernels used | O128 to held-out confirmation and exact patch; O256 not resolved beyond noise | `f22020d25d234be6` |
| Owned VDSK pool128 S3,128×128×64,st5/w8,TMA | L/no-pool128 .998126 at v16 | FP32 partials feed original route with PDL release/wait | Small tentative signal; confirm pool alone and beside O128/PDL | `f22020d25d234be6` |
| Owned VDSK unpool128 S2 | L/no-unpool128 1.003428 at v16 | FP32 partials feed original boundary2; TASK passes | Close this tile/split recipe as a win; other owned unpool designs remain open | `f22020d25d234be6` |
| Native N128 VQ8 st4/mt2 at v1 full | L/no-v1full 1.007972 at v1 | Every full key/current token retained; TASK passes | Close this FA3 replacement recipe in the full layer; alternative native designs remain open | `f22020d25d234be6` |
| Native N128 VQ8 st4/mt2 at v32 sliding | L/no-v32slide 1.006421 at v32 | Every window key retained; TASK passes; increased seed spread | Drop this replacement; no claim that all native sliding loses | `f22020d25d234be6` |

| S15 vs range-dispatch S15n vs SGLang,65 shapes | Nearby/scored speedup by d1/d16/d32/v1/v16/v32: S15 .9547/.8651/.8184/.8119/.7327/.7263; S15n1.0292/1.0142/.8884/.8682/1.0050/.9360 | Complete2seeds×8coldreplays, SGLang available, finite/schema gates pass; many unknown-B calls otherwise lose specialized MoE/attention/projection/glue. Same scored dispatch control ranges .97794–1.03416/.99554–1.00231/.99239–.99919/.99428–1.00065/.99491–1.00196/1.00054–1.00283; no independent A:A and slight d32/v32 drift | Large nearby cliffs improve; no scored gain claim or correctness promotion. Native TASK67-shape validation of guarded S15n9cc90d2e sent; B<=64 guard preserves all65 timed cases. Other neighboring contexts/batches and final integration remain open | `a793c99eb9d447df` |

## Projections and verify dense

Current small-M: balanced split-K Triton GEMVs, lossless coded qkv/pool_router at t1,
pool_router bal2 BK128/2 stages at t32; O config 6 at t1; larger M pinned cuBLAS/Lt.
FP32 router is hi+lo BF16 products into FP32. SGLang dense fit 2.8 us +bytes/3.54 TB/s;
router 10.3-26.3 us vs 1.8-2.7 floor (`7fb3e930`).

Balanced geometry: qkv S7/252 CTAs, pool_router S6/432, unpooler S2, BN64/BK128, 4 stages,
2 CTAs/SM/ ~80 KB. Paired chains save 3.2 /4.2 /7.1 us per layer at t1/16/32 (`5cefdbc4`).
Roofs (`9cd95e72`, read flush): plain reads qkv 10.7-10.9 us, pool 18.7-19.1, unpool 17.0,
slice 4.1, O 5.8. Balanced qkv 11.62-11.70 /11.81-11.92 /12.26-12.35 at t1/16/32,
pool 20.94-21.18 /21.09-21.33 /21.98-22.05 (`26fd09a7`,`a1927852`):
 ~0.6-1.4 us qkv and ~2-3 pool left to a plain read. Entry-to-issue 47-101 cycles,
PDL wait 10-11, marker-to-issue 0.51-0.77 us (`95ab383d`): mainloop/reuse dominates.

Gate GEMV alone 6.5-7.1 us stretches beside attention: 6.7 at d1S, 18.3 at d1F, 96.4 at d16F
(`0a48a058`). Overlap saves bandwidth attention leaves idle, chiefly short latency-bound cases.
Verify-dense max(read, compute) roof would save ~12 us/v16 and ~10/v32 ( ~+1% score);
M128 HBM-bound, M256 qkv/pool/unpool tensor-bound, O narrowly HBM-bound (model 989 TFLOP/s).
cuBLAS split-K M128 qkv main 12.5 +reduce 3.1 us; router mains 9.0 +reductions 3.9 +add 1.1
(`458c6d3e`). Few output tiles underfill 132 SMs; fusion can also destroy that parallelism.

Public NF v18 (`86aff75`, MIT) port at S14 call sites (`e3c637d1`, port/ours):
qkv 0.707 at t1, 0.96 at t8, 0.817 at t32; O 0.849 /0.930 /0.927 /0.961 at t1/8/16/32,
0.91-0.96 at 128/256; unpool 0.808 /0.900 at 16/32, 0.97-0.99 elsewhere;
fused pool/router loses 1.07-2.49. Estimated layer+1.5-2%, whole-layer pending S15.
Same 12-bit exact encoding as ours; structural difference: weights on MMA M,
`ld.global.nc.L1::no_allocate`, cluster reduction, pinned large-M Lt.
Live evidence `problems/muse_bf16_proj/dev/public_nf/`.

PDL audit (`8db42132`,`0f74021a`): 17/19 cluster specializations compile intended preloads
after wait; fixing all loses 0.885 speedup ratio. Register-hoisted tiles grow 48→150 registers,
LDGSTS 40→8; hints 1.0027, metadata 1.0032 (A:A 1.0007), tiles 0.887. Confirmation pending;
`problems/muse_bf16_proj/dev/pdl_audit/`.

### P10/P12 reset measurements

Recipe ratios are candidate/control (lower is faster), except the explicitly labelled component speedups. Full shape/seed evidence and code pins: `problems/muse_bf16_proj/dev/sparse_all/results.json` and `problems/muse_decode_layer/dev/p12_components/leader_gap.json`.

| Recipe | Ratio and control spread | Mechanism | Leaves open | Receipt job |
|---|---|---|---|---|
| P12 S15 projection component | SGLang/ours 0.98059 [0.96588,0.99530]; ours/leader 1.65590; A:A unavailable | Frozen S15 path behind the board ABI; official paired SGLang control | 9 tests/30 shapes pass. Leader per-shape times unpublished; aggregate gap only; no layer promotion. Includes board sum/cast/router-add tax; one-reducer/producer-only follow-up queued. | d6ad694a27954c80 |
| P12 S15 moe component | SGLang/ours 2.02924 [1.99880,2.05968]; ours/leader 1.25835; A:A unavailable | Frozen S15 path behind the board ABI; official paired SGLang control | 5 tests/6 shapes pass. Leader per-shape times unpublished; aggregate gap only; no layer promotion. | 97f5e7f3c3fc4159 |
| P10 pool sparse batch | unmeasured: failed or incomplete control/gate | No recipe loss or promotion inferred | LAB_ERROR CompilationError: at 6:4:; AttributeError("'NoneType' object has no attribute 'startswith'"); AttributeError("'NoneType' object has no attribute 'startswith'"); LAB_ERROR host supplies weights=None; payloads needing weights must synthesize them (state is also None on import-time hosts) | b02c4f27e71c4340 |
| P10 native reader T1 o | scalar 2.8559; vector 1.9388; hoisted 1.7258 [1.6875,1.7769]; AA 0.0509 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T1 pool_router | scalar 2.9181; vector 2.9461; hoisted 1.5186 [1.4856,1.5405]; AA 0.0180 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T1 qkv3 | scalar 2.8628; vector 2.6410; hoisted 2.0212 [1.9836,2.0774]; AA 0.0307 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T1 qkv_gate | scalar 2.4827; vector 2.4379; hoisted 1.5906 [1.5436,1.6078]; AA 0.0309 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T8 o | scalar 3.1813; vector 2.1402; hoisted 2.1710 [2.1127,2.2551]; AA 0.0261 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T16 o | scalar 2.7611; vector 2.6437; hoisted 1.8668 [1.7843,1.9187]; AA 0.0434 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T16 unpooler | scalar 2.4719; vector 2.0911; hoisted 1.8947 [1.8671,1.9123]; AA 0.0177 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T32 qkv3 | scalar 2.2702; vector 2.0040; hoisted 1.6462 [1.6212,1.6732]; AA 0.0301 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T32 qkv_gate | scalar 2.7584; vector 2.4959; hoisted 1.6404 [1.6282,1.6740]; AA 0.0259 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 native reader T32 unpooler | scalar 3.4319; vector 3.4155; hoisted 1.9563 [1.9371,2.0129]; AA 0.0271 | Identical bank/escapes: scalar vs parallel butterfly/PRMT vs invariant pointers plus independent dense loads; raw12/S15 controls | Measured loss closes these three indexed sparse readers on this stage; alternative transport/layout and stronger compression remain open | 3a91afeed9ea47fd |
| P10 grouped Triton point reader | unmeasured: T256 fused-route equality gate failed | Grouped codewords/masks per1024W inside original dot loops | Native ten-cell section is independently gated and valid; grouped section needs membership/order/tie diagnosis and a fresh gated run | 3a91afeed9ea47fd |
<!-- P10/P12 measurements end -->

## Glue and routing

Current t<=32: DSM cluster of 8 CTAs/token x 256 threads with PDL; row slices reduce norms,
route uses integer-order max keys/top8, not sort. Paired chain gains b1 -1.01 /-0.53 /-1.14 /-0.53,
b2 -1.42 /-1.04 /-1.71 /-0.61, route -0.26 /-0.38 /-0.37 /-1.41 us at t1/8/16/32
(`1365a290`). Larger T uses row-per-CTA Triton; load weight vectors after wait at t256:
b1/b2 -1.07 /-1.13 us (`3345699f`).
BF16 front replaces torch sum+cast with one PDL Triton reduction; attention launches normally:
chain -4.21 /-4.46 /-5.25 us at d1S/d1F/d16F (`87cd3a84`).
Early attention release lets O config 6 load before waiting: d1 S/F -0.95 /-2.06 us
(`4d995f13`); more PDL is not automatically more overlap.
`MoERouteMeta` v1 is 2,944 B int32; cluster producer's arrival counter resets in-call,
last token rank 0 writes without spinning; all 711 field words match at t1..32 locally.
Boundary critical-path attribution: `problems/muse_bf16_proj/dev/boundary_attribution/`.

Reset batches below are **unmeasured**, not losses: the prefill receipts contain no LAB diagnostics; displacement was not established. Host submission scores do not measure the embedded recipes. Exact payloads were requeued on the attention host, whose tuning precedes the lab.

| Recipe | Ratio vs control / A:A | Mechanism established | Leaves open | Receipt |
|---|---|---|---|---|
| S14 metadata confirmation + d16/d32 boundary attribution (16 seeds) | Unavailable / unavailable | No LAB diagnostic; host correctness passes | Every tested preload, geometry, serialization and MoE-release/unpool-hint arm; exact retry `ce5b991a8a984a03` (sender 4131535) | `4c86853a953b4eca` |
| S14 decode-1 b1/route/b2 preload chain (11 arms, 32 seeds) | Unavailable / unavailable | No LAB diagnostic; wrapper mechanism unresolved | W/W1/WR/W2, late-prew, hints, early residual and CL4; exact retry `5a95984a640d410e` (sender 4131717) | `841077efefb343a7` |
| S14 d16/d32/v16/v32 boundary rebuild (10 arms, 24 seeds) | Unavailable / unavailable | No LAB diagnostic; wrapper mechanism unresolved | Moment/exact single-CTA, serialization, route and tile-publication recipes; exact retry `9f6849fcc12a448d` (sender 4131824) | `a05f986396d744e7` |
| S15 all-stream critical timeline (12 shapes, 3 seeds, 108 profiles) | Mapped/plain 0.97566–1.02431; profiled/mapped 0.99639–1.01627; all gates/fingerprints pass, accounting error 0 | Verify DAG gaps overlap heavy/light MoE work: representative v16/v32 busy overlap 28.14/105.56 us, GPU idle only 5.45/5.86 us. Decoder separates them; these are not CPU launch gaps | Glue above fitted stream roof 3.6/0.5/0.5/0.7/1.3/1.5% at d1/d16/d32/v1/v16/v32 (median-seed owner traces); no kernel promotion. Full table `dev/s15_timeline/H200.md`; model includes 2.36 us intercept, unlike byte-only PLAN | `7eaa262a30ec483e` |

### Retry decisions and MoE shell hand-off

Ratios are candidate/control latency; below one wins. A:A is an independent capture of the same S14 handles. Each row states the remaining scope rather than closing a kernel family.

| Recipe | Ratio vs control; A:A | Mechanism / decision | Leaves open | Receipt job |
|---|---|---|---|---|
| Exact metadata confirmation + d16/d32 attribution retry | Unmeasured; no retained ratios/A:A | LAB_ERROR: combined archive exceeds 9 KB. No promotion or performance closure | Metadata-only preload confirmation and original attribution arms; needs compact diagnostics, not fewer controls/seeds | `ce5b991a8a984a03` |
| d1 glue chain, 32 seeds | W S/F 0.98648/1.00369; D full 0.98549; A:A S/F **0.98549±0.00313 / 1.01603±0.00539** (95% mean CI) | Gates pass, but control drift is comparable to apparent gains; no qualified win or family closure. C4 observed 1.03119/1.02371, hints 1.00879/1.01158 | Same-capture balanced AB/BA confirmation of preload ownership; source of A:A bias is not isolated | `5a95984a640d410e` |
| Full-row moment/exact, serial, route CTA and chain at d16/d32/v16/v32; 24 seeds | M **1.07236/1.05352/1.01380/1.00461**; E **1.07590/1.05995/1.01794/1.00843**; R **1.05884/1.07432/1.00838/1.00409**. A:A **0.99813/1.00069/1.00392/0.99991** | Tested clean-routing rows lose beyond A:A. d16S ready b1 A/M/E 7.168/13.712/13.712 us; route A/R 8.688/16.992 us. One d16F row changes membership for M/E/S/C/E1, so those rows remain unmeasured for promotion | Other CTA geometries/producer ownership; E1 v16 is inconclusive; changed-routing d16F rows require official margin review | `9f6849fcc12a448d` |
| Decode per-tile unpool publication + exact b2 | d16/d32 **1.08951/1.07456**, same A:A as above | Close this producer/consumer recipe: sampled b2 completion tail 2.753→15.488 / 3.167→14.304 us. Verify P is an unchanged duplicate control, not a publication test | Other readiness protocols, faster current producers, and finish ownership; no general publication/fusion closure | `9f6849fcc12a448d` |

## Weight encoding

Expert banks remain packed E2M1 + uint8 E4M3 recodes; scale sign/zero folded into codes.
BF16 scale words built in prepare are widening even if represented by int16: forbidden;
`ad858b58`,`e2f45792`,`fecc29af` withdrawn. Form words `(code<<4)+0x7000` in registers.
All 4,064 E2M1×E4M3 products fit BF16 exactly. Synthetic decoder 134M weights:
4-LOP3 0.98 Tweights/s vs PRMT 1.89, excluding scales/packing/MMA (`849bd4b6`).
Dense weights' exponent entropy 2.545 bits (4M randn samples): lossless ideal 10.54 bits/weight;
16 contiguous exponents cover 99.9928%, 8 cover 98.64%, 4 cover 85.5%. Exact 12-bit representation
cuts 25% ( ~34.6 MB of decode 1 ~138 MB dense reads), with escapes. Coded integer decoder 1.125
instructions/weight ( ~21% of issue slots at 4.58 TB/s), not issue-bound; token-dependent
activation staging and occupancy caused larger-M losses (`32088444`).
Direct-BF16 decoder reconstructs high byte + even exponent origin without per-output multiply;
standalone wins at t1 do not establish whole-layer gains (`a645a795`, S14 rejection below).

## Closed recipes

Ratios are candidate/control time unless explicitly called speedup. Deltas/absolute pairs remain
as measured where no ratio was recorded; no ratios have been re-derived. "Not isolated" does
not close a family; incorrect/untimed implementations are suspect or unmeasured.

| Exact recipe | Measured result | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Hiding a stage's ramp behind its predecessor (weights in registers / L2 before `griddepcontrol.wait`, bulk L2 prefetch) at attention -> O and O -> b1 -> pool_router | Every recipe loses (+0.1 to +10.8 us, `5cefdbc4`, `cc718d3f`); the oracle ceilings with the bytes already warm are <= 0.7 us (attention -> O) and <= 2.0 us (pool_router) | The 2.36 us cold-read intercept (`7b6c96b5`) is mostly launch, flush and tail, not latency that a prefetch hides; prefetches compete with the running stage for HBM and issue | Unpooler at t=16/32 only by reading its bytes during an HBM-idle predecessor (none at small M) | `5cefdbc4`, `cc718d3f` |
| Skip pool/unpool rows without local experts | Local no-route medians 13.3%/14.8% at verify 16/32; range 1.6-24.6%; H200 unmeasured | Still reads weights; compaction+router dependency; estimated <=2-4 us/verify layer | A costed compact-row implementation | local; commit `3400869` |
| Router-epilogue L2 prefetch of the MoE's first W13 tiles (M=1, 13.9 MB) | Router node 4.1 -> 51.2 us; router+MoE 53.2 -> 92.2 us (local sm_120, read flush) | One CTA's bulk prefetches run at a single SM's TMA rate and hold the CTA's exit, so the dependent MoE waits | Weight prefetch issued from many SMs | local; `problems/muse_decode_layer/dev/nt1_timeline/README.md` |
| MoE M=1 streaming expert GEMV (persistent W13 rows + W2 split-K, fused norm/sum, 14 grid/stage/PDL variants) | Best P2g264 57.07 us vs NT1 44.17 (1.29x slower), 12 seeds; PDL and grids 264-528 barely matter, split8 95 us | Not isolated; NT1 already reads 1.4-1.7 TB/s unique and the recode alone gives 9-12%, so per-weight decode work, not bytes in flight alone, binds at small M (inference) | Streaming with a cheaper decoder | `963515d3` |
| MoE NT2 / NT4 token blocks at M=1-32 | NT2 +16% at M=1, +28% at M=8; NT4 +63-96%; NT4 149.2 / 141.4 vs NT1 93.7 / 110.8 us at M=16/32 | NT4 spills (288-byte frame, 500 / 344 spill stores / loads, static); NT2 not isolated | NT4 without spills | `b68afb72`, `b741f04a` |
| MoE LOP3 NT1 switches: A metadata pass, B expert-local ready/norm/combine, C W2 L2 lookahead, K2 half tiles/grids 528/396 | A+1.4-1.7 us; B+0.3/+2.7/+16.4/+30.4 at M1/8/16/32; C+1.7-9.2; K2+45-66% | Not isolated | Parallel producer metadata; SLOT already admitted | `383ccc1f` |
| Router serial epilogue writes MoERouteMeta, MoE reads | Router+MoE+3.24/+2.94/+3.12/+3.27 us at M1/8/16/32 | Serial metadata costs more than per-CTA rebuild | Parallel metadata producer | `849e4017`,`28fc0380` |
| M1 last-arriving CTA runs finalizer | +0.11 us alone, +0.36 with SLOT; all flags+LAST -5.71 vs -5.72 us router+MoE | Serial finalizer offsets removed release | Distributed finalizer without release | `849e4017` |
| MoE producer/consumer split D64/D128 at M=16/32 (128 producer threads decoding into double BF16 smem for 256 MMA threads) | 189.6-230.5 / 299.1-364.5 us vs 79.7 / 114.1 (2.4-3.2x) | Implementation suspect (no phase evidence) | Heavy experts only: lost at M=128 (next row) | `383ccc1f` |
| MoE decode-once producer/consumer for heavy experts at M=128 (128 producer threads decode each expert/channel/K64 tile once into B128-swizzled smem; 128 consumer threads issue SS-WGMMA per token subgroup; 2/3-slot mbarrier ring) | vs S7's H_H 232.99 us (read flush, 12 seeds x 3, gates pass): best B64s2 1.1305, T32_A64 (only > 32-row experts) 1.1500, A64 1.26, P64/P128 1.39, P32 1.46; heavy W13 phase 55.6 -> 97-114 us; at M=16/32 2.2-2.8x NT1; grid 528/1056/2112, N width and ring depth barely move it | Not isolated (source-audit candidates: no activation register reuse across channel tiles, the extra smem round trip, per-stage full/empty handshakes) | An `mma.sync`-from-smem consumer (adds ldmatrix traffic); decode-once without the smem round trip | `9cd95e72` |
| MoE NT1 joint E2M1x E4M3 table decoders (sm_90a K-loop, NT=1 g-loop) | Static SASS per 16 weights, 4 HMMA per 32 weights excluded: LOP3 26.06 (= lop26); 8 KiB code x scale BF16 table 90.6; 1 KiB code-pair table + HMUL2 34.3; a 256 KiB code-pair x scale table does not fit shared memory | Per weight pair the table costs 2 LDS + LOP3/SHF/LEA index math + a PRMT pack, more than LOP3's bit decode; screened by static instruction count (not an H200 family closure) | The bank format; decoders other than table lookups | local SASS; `problems/muse_decode_layer/dev/nt1_timeline/table_kill/` (`ktable.cu`, `count.py`) |
| MoE V9 with PRMT NVFP4 decode on V9's bit-plane layout | Bit-exact (524,288 cases); local 1.126x vs V9 1.283x, V32 501 -> 703 us | Rebuilding PRMT selectors on V9's layout grows the sm_90a inner loop 492 -> 1,255 instructions (LOP3 162 -> 545, SHF 112 -> 368) | A layout packed for PRMT in prepare | local; commit `6652e48` |
| MoE M=128 one-read expert-wide RS / shared-A SS (one pass over each touched expert) | 361.2-786 us vs NT1-recoded 290.8 (RS64g2112 361.16 best); 36/36 gates; logical packed-weight repeats 1.000 vs NT1's 2.272 | It changed tiles, addressing and occupancy along with reads (seed0 W13 236.1 vs 132.9 us in H_H), so it doesn't isolate what rereads cost | Fewer rereads inside the H_H structure | `640d67e7`, `770809cf` |
| MoE BF16-resident hybrid (DeepGEMM BF16 + V9 cold experts, bank dequantized in prepare) | M=256 337.2 vs 396.1 us; decode run 1.461x, withdrawn | Inadmissible (CONSTRAINTS 5); the gain is work moved out of the timed call (BF16 for 21 experts would read ~2.1 GB, ~440 us, vs ~124 packed) | Nothing: inadmissible by rule, not by speed | `b37b93de`, `07d07d96` |
| Hardware generic memory compression (cuMemCreate compressible, granted on the judge) | 256 MiB cold read: zeros 2.03x, pattern 1.70x, random 0.997x, raw NVFP4 0.999x, recoded NVFP4 0.998x, BF16 1.002x; NT1 M=1/8 and qkv/o M=1 0.9996-1.0025 | Real weights are incompressible to the hardware (random-like bytes) | Software coding of weights in our own kernels | `4233660b` |
| Attention FlashInfer 0.7 page-1, full context | q1 tensor-core decode 45.5 / 195.2 / 371.9 us at B1/16/32 (1.68-1.76 TB/s) vs `attn_main` 108.0 / 199.7 at B16/32; q8 fa2 prefill 60.4 / 469.1 / 912.3 vs FA3's earlier 208 / 390. Errors 0.0023-0.0027; plan 0.2-0.3 ms outside the call | Unpaired: the lab's own FA3 control failed (dtype); not isolated | Sliding cases (FlashInfer asserted on the window), its fa3 backend (JIT build failed), XQA (needs page >= 16) | `8ffbac2d` |
| Attention q1 FP8-KV split with cluster split reduction | Full B16 / B32 237.8 / 388.1 vs `attn_main` 118.9 / 226.0 us; sliding within ~1 us | Not isolated | Other split reductions | `36672667` |
| Attention q1 conditional O/l rescale (warp-uniform skip when no row max changes) | Profiles full 15.6 / 105.3 / 197.3 vs 15.5 / 103.6 / 193.3 us; sliding 9.3 / 11.6 / 15.0 vs 9.0 / 11.5 / 15.1 | No gain | Rescale skipping at q8, where max changes are rarer per tile (unmeasured) | `9828bfa0` |
| Attention q8 deferred denominator | Full 428.0 / 739.8 us on the emulated `verify8_main`; sliding 68.6 / 83.6 on xqa_group vs FA3-based 23.9 / 30.8 | Unmeasured: confounded with families that lose for other reasons | On the native WGMMA producer | `9828bfa0` |
| Attention q8 K64 / K128 swap-AB XQA (keys on M, raw E4M3 QK, FP16 or FP8 PV, 3-5-stage page-row gather, last-CTA merge) | Full B16 / B32 616.7 / 1173.8 vs FA3 207.4 / 391.4 us (0.336x / 0.333x); sliding 0.51-0.68x | Issue / softmax / layout overhead: 0.53-0.56 TB/s logical; stages 3-5 equal (not gather latency) | Swap-AB with native FP8 WGMMA and gather warps | `32c5b4b2`, `a0511c00` |
| Attention q8 FP8 P and V, one 128-thread CTA per KV head and split, last-CTA combine | 1.25-2.37 ms; a parallel fast combine changes almost nothing (1,271.8 -> 1,255.4 us at B16 full) | Per 64-key tile, serialized: QK ~570, softmax 2,200-2,700, PV 2,800-3,100 cycles; 0.13-0.15 TB/s | Parallel per-tile producers (the native WGMMA row) | `948044d4`, `0812f097` |
| Attention q8 packed-port verify kernel (two-wave plan, 17 / 9 splits) | Full 412.5 / 761.1 us; post-store gate fails (1.252) | Odd split counts break its q8 reduction path; combine 38-44K cycles vs 0.6-1.4K for QK/softmax/PV | Even split counts (8 / 4 pass locally) | `b02a42ce`, `0467893d` |
| q8 FlashDecoding verify 8_main, FP8 mma.sync, 17 splits/3 stages/128 keys | B16/32 producer 427.8/713.7 us (0.77/0.92 TB/s), combine 6.3/8.5 vs FA3 ~206/391 call | Emulated FP8;223 registers limit 2 CTAs/SM | Native WGMMA | `be19da94` |
| Attention q8 native FP8 WGMMA Q-major M64, splits 9-33, 4/8 stages | Best 275.1 / 524.3 us (33 splits) vs FA3 210.8 / 395.1; gates pass | Consumer tile dominated by V transpose + barrier ( ~990 cycles) and softmax ( ~760); ring depth not binding (8 = 4 stages); more splits help monotonically | Moving V transpose and softmax work off the critical warps; > 33 splits; more resident CTAs | `a1679f9c` |
| Attention FA3 scheduler metadata on a side stream at verify (`get_scheduler_metadata` forked at the call start, joined before FA3; bitwise-equal output) | Chain qkv -> q FP8 -> FA3 -> consumer +0.82 / +0.08 / +0.99 / -0.46 / +1.31 / +0.02 us at verify 1 / 16 / 32, sliding / full | Inferred: `prepare_varlen_num_blocks` (2.1-3.6 us in the profile) is not exposed in the chain, and the fork/join costs ~1 us | Nothing at this step | `e1a56ee7` |
| Attention split partials merged through global memory instead of a DSMEM cluster | 1.190x vs v2 1.279x | An extra global round trip per split | Fewer splits, or long full-layer contexts | `8bfd5e25` |
| Proj verify dense weight-wide TMA/WGMMA, regular tiled split-K (M=128/256) | 137.4 / 159.9 us summed vs cuBLAS/Lt 85.3 / 100.4; every split-K > 1 lost to split 1; also 1.28-2.18x in `5535d2f1` | Not isolated | Triton split-K at M=128 qkv (next row); cuBLAS itself reads only 1.7-2.4 TB/s | `19f3a13b`, `5535d2f1` |
| Proj verify dense Triton WGMMA split-K (FP32 partials, S=3) beyond M=128 qkv | M=128 pool_router / unpooler / o 1.03-1.09x and every M=256 role 1.03-1.19x of cuBLAS; DeepGEMM 1.00-1.17x there | Not isolated (inferred: wave fill; qkv's N=2304x M=128 has few output tiles, and S=3 triples the CTAs only where that helps) | M=128 qkv: 0.8777 ("Projections and verify dense") | `b15a3bde` |
| Native bulk-ring GEMVs M1-32, standalone and O→boundary→pool/router | Standalone 1.03-1.73x; M8/16/32 chains 1.04-1.64x; M1 O6+ringP 0.9508, -2.1 of 42.8 us | Not isolated; write-flush adds dirty writeback, M1 win suspect | M1 under read flush | `87b2667f` |
| Proj verify custom TMA v2 | M=256 qkv 23.49 -> 37.73, pool 33.39 -> 65.55, unpooler 28.19 -> 44.85, o 13.34 -> 20.51 us; M=128 similar; PDL chain 1.32-1.62x | Not isolated | Other custom TMA tilings | `92e78027` |
| Proj verify stock CUTLASS Hopper cooperative (144 recipes per key: split-K, Stream-K, multicast, swap-AB, rasters) | Every choice loses: M=256 1.08-1.31x, M=128 1.14-1.32x of cuBLAS/Lt | Not isolated | Custom CUTLASS kernels outside the stock schedules | `45f58e70` |
| Proj register-MMA unsplit GEMVs (14 kernels) at M=1-32 | 15 of 16 role finalists lose; O -> boundary -> pool segments lose 5.37% / 20.97% / 70.91% at M=8/16/32 | Prologue and PDL wait are small (47-101 / 10-11 cycles); loss is mainloop transport and reuse | M=1 O config 6 (kept) | `95ab383d` |
| Proj original K64 TMA/WGMMA, one / two consumer warpgroups, fixed DSMEM dispatch | 0.820x / 0.824x; m128 1.3-1.8x, m256 1.7-2.1x slower than cuBLAS | Unbalanced output waves, per-K64 synchronization, cluster merge overhead | Larger K tiles, Stream-K merge, per-shape tuning | `87dd70d5` |
| Proj skinny `mma.sync` split-K with cluster DSMEM reduce, m <= 64 | 0.991x; qkv / o 0.55-0.83x at m=1/8/32 | Split-K and cluster-reduce fixed costs | In-kernel last-CTA reduce | `03113fd3` |
| Proj cuBLAS split-K pool/router replacing custom GEMV at small M (union) | +6.6 / 7.8 us at decode 1 S/F, +8.9 / 12.6 at decode 16, +11.2 / 17.0 at verify 1 | One extra cuBLAS kernel (22 us) + a split-K reduction (1.7 us) in the graph | cuBLAS at M >= 128, where it already runs | `215cafaf` |
| Persistent O→boundary→pool/router, grid barriers/L2 hints | M1/8/16/32/128/256 57.7/62.7/65.3/71.2/198.9/365.6 vs 38.3/39.2/39.4/39.8/43.6/56.3 us; +21 us M1 layer | Fixed NT4 padding,132-CTA serial task trips, 3 grid barriers, 4/8x bank rereads at M128/256; no retained operand ring (static) | Tile readiness with competitive inner loops/retained operands | `87dc944b` |
| Persistent qkv chain (`commit 9d8785c`) | Rejects at (32, 1) nondeterministically; 0.86% hidden error and changed expert IDs at (8, 8) | Implementation suspect (launch-order race) | A deterministic persistent qkv producer | local |
| Glue fused into streaming GEMVs (norms, residual, routing) | 2.3-21x cuBLAS (`5535d2f1`); every fused stage, prefix and chain 1.04-11.4x (`479aab30`) | Folding glue into skinny GEMVs costs their parallelism | Glue fused into consumers | `5535d2f1`, `479aab30` |
| Cluster glue at t = 128 / 256 | Every cluster shape 1.02-1.28x of S6's Triton glue | Not isolated | t <= 32, where it wins ("Glue and routing") | `1365a290` |
| Glue small-M fusion, prologue and last-CTA epilogue cells | 1.51-17.63x slower | Implementation bug: the pool grid recomputes each 8192-column norm 576 times per token; last-CTA tails serialize on 1/1/2/4 CTAs at M=1/8/16/32 | Repaired cells (next row) | `80b70f7f` |
| Glue cooperative resident grid: one RMS/row, 8 tails/row, route-once v1 | Pool 1.40-5.67x, tail 1.16-5.91x, all 1.65-9.53x, epi 1.58-9.93x, route-once 1.02-5.53x; 12 cold cases, gates pass | Grid-wide barrier/row suspected, not isolated | Fusion without grid-wide sync | `9eeb39ba` |
| Side-stream L2 prefetch beside the layer | Layer 0.3-16.6% slower (decode 1 sliding 123.1 -> 143.5 us); branch ~34 us | A separate launch can't run beside full-grid kernels (it overlapped attention only, never MoE) | In-kernel prefetch | `d5a90258` |
| In-kernel PDL own-weight prefetch | d1S 115.78→115.58 us, 0.9986; roles 0.9910-1.0094 | 7/8 cases unmeasured: own 0.003 gate rejected 0.0031-0.0033 | Other cases; no family closure | `7426e17c` |
| M1 global per-unit DYN counter (1119 vs 95) | -1.78 +-0.48 vs-5.68 +-0.11 us; W2 wait critical 5.92→2.42, W13 end 14.5→18.5 | L2 atomic ahead of each 1-2 tile copies | Coarse assignment without per-unit atomic | `eb9e0382` |
| Decode-once registers: NT2 203 registers/1 CTA, or W16 123 registers at M16/32 | NT2+11.8/+7.9 us, W16+19.0/+17.2 vs INC; flags 69 -4.7/-4.5 | NT2 unit 1.84x cost, 16-token register activations prevent 2 CTAs | Reuse preserving residency | `c768a8e3`,`daf0c1aa` |
| RND/ROT two expert rounds; WPUB independent writers | RND+0.0/+6.7/+6.5 us M8/16/32; ROT+3.1/+4.2/+7.9; WPUB+1.0/+1.1/+0.4 | M32 ready 11.2→2.1 but barrier 4.3→12.7, W13 37.4→41.1 | Finer true dependencies without moved wait | `17512d86`,`3af06b40`,`4dd00540` |
| Unique-weight ONCE vs REDO over all blocks | ONCE-14.75 us at M32 vs twin; vs flags ONCE 1.30/1.30/1.33, REDO 1.13/1.26/1.46 M8/16/32 | Ownership/activation streaming exceeds saved decode | Reuse with original activation efficiency | `e4ad355e` |
| Q4 queue; BUNDLE/COMPACT/prefetch; ACT/DECODED small-M reuse | M8/16/32 Q4 1.567/1.559/1.260; BUNDLE 1.118/1.162/1.241, COMPACT 1.127/1.130/1.250, prefetch 1.130/1.207/1.383; ACT 1.163/1.319/1.782, DECODED 1.206/1.357/1.785 | Extra-block gathers replace register reuse; 58.7 us math/activation, 28.5 epilogues vs 3.6 weight waits | Ownership preserving activation reuse | `a2e4cf10`,`b5659458`,`2fad2371` |
| Register-A WGMMA 64 x 8/16/32, decode once, no K-split | 1.52-2.29x; 1.0-1.9 TB/s | K-split supplies bytes in flight at small M | Reuse with enough concurrent requests | `00f4adc3` |
| N_full co-resident heavy grid 2/3 CTAs/SM | M128 1.024/1.029, M256 1.059/1.043; 1 CTA 1.24/1.40 | Heavy needs every SM's bandwidth | Branch-local pipeline | `f976b09d` |
| Heavy wide banks/typed activation/two accumulators/adaptive groups | Wide 0.9924/0.9884, typed+wide 0.9918/0.9857; two 1.0225/0.9794 (0.90-1.13); adaptive M128 1.015-1.16 | Smaller static count not reliably faster; not isolated | New schedules; PERMW admitted instead | `89608c59`,`b88d04fe` |
| N16G4 shared queue: phased or expert-ready persistent DAG | Phased 1.034/1.037, DAG 1.100/1.107 at M128/256, all pass | Not isolated | Lower-overhead readiness/ownership | `ea794250` |
| Expert-wide cluster activation sharing, decode once | vs S12 CS1 1.319/1.252, CS4 multicast 1.322/1.260; vs S14 TMA 1.404/1.314, lean 1.439/1.334, parallel 1.230/1.150, direct 1.197/1.132 | Correct/bit-equal; reuse did not repay new transport/ownership, not isolated | Another ownership with cheaper activation path | `5fdb668c`,`8ee003db` |
| Cooperative four-lane QK prep+vq8 merge/shared RoPE | 1.07-1.39 sliding, 1.22-1.31 decode full, 1.03-1.09 verify full; revised 1.08-1.30/1.25-1.27 decode | Not isolated | Independent history gather | `8fd0d2a6`,`8b8e4e2b` |
| Single-launch DSM sliding; 32-thread m16 FP8-MMA/all page IDs in flight | d1 DSM 0.996-1.011, d16/32 1.02-1.10; m16 d1 1.01-1.18, d16/32 1.03-1.13 | History gather 5.3 us dominates; dependent chain persists | Early history staging, last-CTA | `79061182`,`c0fd644c` |
| Balanced native vq8 softmax; FP32 polynomial exp | Native 1.005-1.020; poly degree 2/3 v16 1.080/1.087, v32 1.105/1.114 | Producer ends ~3 us before consumer; softmax replacement not faster | Different softmax dependency/issue schedule | `8b8e4e2b`,`9407b461` |
| Full-attention more page flight, FP16 PV, register-V, CTA160, LDSM.T N64 | Flight 0.997/0.997; PV 1.50/1.55; scalarV 1.43/1.47; rawV 1.23/1.26, P-register 1.06/1.11; LDSM 1.09/1.12 | Not isolated; vq8 softmax ~907/ ~1690 cycles/tile | Register-fold N128, other issue layouts | `39949536`,`bdfeb0e9`,`6ec9d43d`,`5cca1296`,`f4d1a195` |
| N128 full verify before register fold/stage 3; register-V | Before fold 1.30/1.45; stage 3 1.048/1.071; registerV 0.996/1.065 | Register/layout/occupancy sensitivity; not isolated | Stage 4 register-fold pending S15 | `7e93cc81`,`9d9a01e1`,`cf79aa07` |
| Fused full-QKV+gated N128; single-head verify; d1 full specialization; FAST/VQ8 verify | Fused 1.003/1.005/0.968 vs split-gate 0.990/0.990/0.968; single-head 1.24/1.31; d1 full 1.39; FAST 1.11-1.61 | Not isolated | Split-gate/two-head designs | `c773699a`,`6d47827c`,`58d6a596`,`2f8b702d` |
| Own Triton TMA verify GEMMs+boundary stats | Whole-layer old-boundary 1.019/1.010, stats 1.042/1.026, O-only 1.009/1.008 at v16/32 | Stats four sums 1.14-1.21, branch-only 1.03-1.06; lost pinned-Lt parity | Different GEMM/epilogue ownership | `d3666d18`,`d4cbf3ca`,`ca685be1` |
| Last-CTA producer glue b1/route/b2; then distributed statistics version | First 1.05-1.11/1.10-1.15/1.03-1.06, all 1.12-1.21; second d1 1.158, metadata 1.177 | Late row-owner tail serializes; d1 b1 4.16→7.52, b2 7.07→13.92 us | Proven pre-wait immutable loads; full-only preloads admitted | `f21c60b0`,`15abb9c1`,`9845398b` |
| Force all intended PDL preloads before wait | Speedup ratio 0.885; register tiles 0.887 vs hints 1.0027/metadata 1.0032 | Registers 48→150, LDGSTS 40→8 | Metadata-only confirmation pending | `8db42132`,`0f74021a` |
| Coded GEMV beyond qkv/pool at t1; all-thread cp.async; direct BF16 whole-layer | Cleanup t16/32 qkv+0.58/+1.57, pool+1.31/+3.19 us; cp.async corrected 0.9969 vs A:A; direct qkv3 off 0.981, bal2 t16 off 0.982, O 0.997 | Activation staging/waves; standalone wins disappear in layer | NF structure and denser exact codes; S15 pending | `3239a2ea`,`4de869d4`,`0441f9fc`,`740c63af` |

## Failures

| Symptom | Cause | Avoid / fix | Evidence |
|---|---|---|---|
| Prefill lab passes at ~1.43x but no LAB markers survive | `--phase import --import-budget 550` printed before the unchanged host tuner; at least 18,818 later host characters displaced diagnostics from the 16,000-character tail, not a prepare timeout (host 113.9–134.5 s; all five compared sources share base `cb8939d0` and one payload) | Buffer import-phase lab diagnostics on hosts with judge-called prepare; emit after host prepare, preserving import execution/deadline and host state | Missing `4c86853a`, `841077ef`, `a05f9863`; working prepare-phase `a39a1f05`, `c773699a`; exact source hashes checked; local judge-image GPU0 proof: 24 wrapper cases pass |
| Lab small-M MoE slower than judge | Write-flush leaves dirty L2 lines; zero_ adds MoE +1.82 / +4.88 / +2.72 / +1.87 us at M1/8/16/32 | Use `lab_cold_l2()` read flush; write-flushed absolute times are ~2-5 us high | `849e4017` |
| Local flush makes lab 5x slower, 30/36 rows finish | uint8 sum over 1 GiB costs 39 ms on sm_120; int32 view 11 ms | Use a wider view; price flush cost in replay-heavy labs | commit `13801d3` |
| Local 22% shared-alignment win becomes 0.6% on H200 | sm_120-only alignment effect, not isolated | Do not rank shared-alignment changes locally | `110599ba` |
| Three scored runs required withdrawal | Prepare stored scale values as BF16-valued int16 words; exactness gates missed illegal widening | Inventory every retained tensor's numerical meaning/precision, not its storage dtype | `ad858b58`, `e2f45792`, `fecc29af` |
| Exact-ID gate calls a slot swap a route flip, aborts lab | Same selected set, experts 172/173 swapped slots 6/7, score delta 2.8e-6, output L2 9.4e-6 | Compare sets; use TASK near-tie rule and report cutoff gaps | `9eeb39ba`, `8a3d7181` |
| H200 router oracle disagrees after local pass | Different FP32 split-K summation order near 8th/9th cutoff (inferred) | Allow documented near-tie flips and report gap/count | `7307be78` |
| MoE agreement error ~1.8-1.9 | Recoded scale bank fed to raw reader: stored `0x70` read as `0xe0` = -32 | Match reader/bank schema; inverse-check bank and first mismatch | `3920d3f0`, `948044d4`; fixed source `fee5f3b0`, job `b741f04a` |
| Humming/V11 loaded-weight error ~28%, synthetic passes | Adapter conflated distinct W13 gate/up global scales | Test actual schema before ranking library | `26d7e05a`; fixed `de6a56fc` |
| Synthetic grouped-MoE winner collapses in layer | Synthetic routing lacks hot experts; six-launch path retains overhead under skew | Use layer router and shared-draft fixtures; require in-layer evidence | `2ad75c19`, `85e67f5d` |
| FA3 verify 32 regresses 377.9→501.5 us | Tuner chose split 32/pack_gqa on wrong length distribution | Tune scored agentic spread or use public scheduler | `691bff0b` |
| Gates skip judge-valid cells or lose all later timings | Incumbent agreement threshold substituted for TASK tolerances | Judge against FP32/TASK with margin; log failing cells, continue independent cells | `aff09765`, `dd927c6c`, `7426e17c` |
| No lab result survives receipt tail | Only last 16,000 characters retained; JIT warnings and ~60 profiler lines displace summary | Quiet JIT logging; compact summary/correctness output | `296fdf45`, `1365a290` |
| WGMMA serializes, C7513 WG.AR/divergence | Live operand copied (`pc=pn`) or nonuniform-looking branch | Two named buffers, unroll 2; broadcast bounds, peel tail; inspect C75xx | commit `d4d2e76` |
| Page/query prefetch pays latency per element | Loaded value consumed inside branch forces wait | Unconditional clamped loads; delay consumption; inspect LDG→BSYNC dependencies | `4aa070c6`, `c816b9a7` |
| Short cold kernel starts slowly | Inlined sincosf/Payne-Hanek and unrolling inflate 17K SASS instructions | Compute sin/cos once/token out of line; compact hot path (7.4K, local query-ready ~4 us) | commit `d4d2e76` |
| vq8's original producer/consumer ring serializes | Consumer holds rr while waiting rr+1; publication waits rr-1 | Publish first, LAG<=ST-2; ST-1 deadlocks; page lookahead 4, not 6-8 | `44aa67ba`, `5928ced8` |
| FA3 control differs from board call | Built explicit `(0,None)` scheduler metadata instead of public q8 field None | Forward `data["scheduler_metadata"]` unchanged | `1b80ffc0`; source `4516cb97`; commit `e8d710f` |
| Side-thread GPU setup breaks graph capture | Global capture mode forbids other-thread GPU work | Finish GPU setup before capture; CPU compilation can run concurrently | source `c582f342`; commit `e8d710f` |
| Prepare reaches 304 s and fails | Lab exceeds 300 s cap | Budget packed payloads below ~280 s | `8ffbac2d` |
| Second payload measures nothing despite time left | 60 s compile gate trips with 57.3 s left | 25 s precompile gate, pack against actual deadline | `14f942d5`; resend `c01b1916` |
| Scored runs misrank sub-3% changes | Independent routes/weights and noisy baseline move common-mode | Paired whole-layer AB/BA on identical inputs; scored run confirms | `a5b2f55b`, `b7c59570`, `ff3b9aca` |
| Judge compilation fails ATen.h despite host pass | Host torch accepts C++17; judge torch requires C++20 | Compile/screen in judge image with `-std=c++20` | `d902de71`; resend `35c875e6` |
| Attention tests 0-5 fail only on some runs | Equal event-quanta times trigger tuple comparison of bool vs None | `sorted(fa3,key=lambda x:x[0])` | `c9c4ef99` |
| Lab uses withdrawn inadmissible host | Host selector chose highest scored run without withdrawal filter | Skip withdrawn hosts | `b63389f0` |
| Send rejected: edits-loaded-modules/starts-processes | Static source checks match sys.modules mutation or prohibited docstring word | Exec payload into plain dict; use sender's REJECTED_WORDS check; read detached log | `runs/detached/731545a5def446ad.log`, `0d7b6cf010101bd1.log` |
| 429 despite reported quota remaining | Accepted-but-unqueued sends count | Sender reroutes to check when available | `runs/detached/053c794fc0aba59a.log` |
| Two files claim same job; one never executed | Sender guessed newest job when submit reply named none | Missing job ID fails send; corrected source attribution is authoritative | `d318167e`, `f7579d53`, `5f402fe4`, `907004cb`, `9f738e22` |
| Send vanishes or leaves orphaned judge job | Caller death kills non-detached send/wait; judge job survives | Detached session; collect by exact job ID, pending receipts | `runs/sent/`, `runs/pending/` |
| Hopper-only glue fails on judge, local passes | Missing CUDAContext include, undefined merged names, bad helper signature, `(partials,split_count)` tuple treated as tensor | Whole-file F821-F823, sm90a-compile all CUDA strings, call signatures; force Hopper Python glue with portable stand-in | `83a8a755`, `43a1f3a3`, `05fe4997`, `2f3960c2`, `69a655d3`, `a5ab4a20`, `b2a89993` |
| Packed candidate exits 112 before tests | CuTe TMA descriptor C++ assert: all-ones dummy dimensions/8-byte scale atom | Real fixed-capacity dimensions, nonzero canonical strides, 16-byte scale atom; host setup before H200/native admission | `a704a37c` |
| CUTLASS cannot implement 11 | Stock RS mainloop requires scale group multiple of K64; NVFP4 group 16 | Call can_implement locally on sm90a module | `3a98cfc8` |
| Single-file candidate imports sibling after local pass | Local candidate directory resolves import, judge has one file | screen_all rejects own-module imports | commit `25d0135` |
| WGMMA K64 outputs nonfinite | af[0] reused while wait_group 1 leaves it live | Alternate register-A banks; audit operand lifetimes | fixed `97be11d4` |
| Persistent projection reinitializes live mbarriers | Persistent tiles repeat mbarrier.init | Finish users, invalidate before reinit; cluster acquire for remote release | `6b72087f` |
| cuBLAS tuner graph faults after another graph resets | PyTorch clears stream-keyed cuBLAS workspace on graph reset | Native stream per tuning graph; reset before stream destruction | commit `c8d91f1`; PyTorch #193402 |
| TMA K128 BF16 descriptor rejected | 256-byte inner box exceeds SWIZZLE_128B | Two K64 transfers/barrier, K64-swizzle WGMMA descriptors | commit `c8d91f1` |
| DeepGEMM dense profile is cuBLAS | Plain BF16 dispatch defaults to cuBLAS | use_deterministic_algorithms(True); inspect kernel names | `84b5433a` |
| Variant scores without its kernel | kernel_fn checked wg_ok absent from prepare state | Store selected branch explicitly | commit `991e838` |
| Broad tuner cold prepare 349 s | Serial compilation 279 s+tuning 70 s | Parallel nvcc/ptxas:122.7 s locally | commit `44b8130` |
| nvcc internal host assertion | `std::vector` heuristic-buffer path | Explicit owned C buffers | commit `1d6a356` |
| Task permutation drops work/deadlocks | `(u*17+3)%total` repeats when total divisible 17; grid 2 over-resides global barrier | Prove bijection and resident barrier grid | `runs/screen/moe_v19_static_audit.json` |
| MoE fails raw-bit A:A | Unordered FP32 norm atomics | Gate by tolerance; slight nondeterminism is expected | `4233660b`, `701d2867` |
| Local ncu ERR_NVGPUCTRPERM | RmProfilingAdminOnly=1, no sudo | nsys, judge profiles, SASS, phase clocks | local |
