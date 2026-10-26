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

### Experiment loop: live closure (P1)

P1 completed in `242e2f2` plus the live closure: S18/S19 share a host-independent promotion evaluator,
per-arm health preserves primary R after isolated optional failures, packets reserve output before logs,
batches use declared compile/run/output/context risk, local preflight precedes prepare/GPU work, decisions
queue by PLAN lane rank, and completed screens send immediately. Archived `5d0282bb` replays identically
through both host paths. Calling contracts and migration remain in RESOURCES "Experiment loop (P1)".

Live check **`1acc6c67671c4220`**, proj host, NVIDIA H200 (10-26; quota before send: 20 checks left): one
resource-planned batch of three small payloads, `--lab-only --check --import-budget 30 --lane P1`.
Receipt and exact submitted source: `runs/muse_bf16_proj/20261026T072505Z_check_1acc6c67671c4220.{json,py}`;
replay manifest and verification: `runs/loop/p1_live_20261026/{manifest,report}.json`.

| Receipt check | Observed |
|---|---|
| Resource planner | One batch, three payloads; reservations 8744 / 1024 / 512 bytes plus markers, 9 s declared work |
| Maximum packet | Exactly 8,744 wire bytes including newline; complete; checksum and padding hash valid |
| Optional fault | Deliberate RuntimeError caught; only this packet incomplete; 508 bytes, checksum valid |
| Subsequent peer | Native CUDA work on H200 completes after the optional failure; 364 bytes, checksum valid |
| Receipt tail | Full 16,000 characters, after 18,000 characters of deliberately noisy import output; all three packets intact; suffix after packets 835 characters |
| Queue and ledger | `lane=P1`, priority 2 (PLAN rank at dispatch), no numeric override; decision `verify live maximum packet, optional failure isolation, and PLAN-ranked batching` |

The minimal lab host deliberately raises at prepare after emitting evidence (`passed=false`); this is the
expected lab-only stop, separate from each packet's completeness. This is a transport/CUDA execution proof,
not a candidate correctness or speedup claim. Kernel promotion still requires the official TASK checker.

Closure fixes: the old 9,000-byte reservation could not fit a maximum packet plus two peers, so lab-only
runs now reserve 12,000 bytes (4,000 left for the prepare-stop suffix; no timed/profile phase); ordinary
transports retain 9,000 and the per-payload maximum remains 8,744. Removing P1 also exposed the implicit
P1 default in old send paths: unlabeled sends now record `lane=unspecified` and follow ranked decisions
once P1 is absent; explicit old numeric priorities remain accepted. `screen_all` and the lab sender preserve
that behavior. Both failures have regression tests. A concurrent import-evidence early/reprint fix is retained: readers
validate every physical copy, collapse identical copies in final emission order, and reject conflicting copies;
a regression covers a receipt that cuts the first large packet. All 25 tests pass in the judge image. Reproduce this proof with `tools/tests/live_loop.py`;
its verifier checks receipt/source hashes, all packet checksums/identities, native device/seed pairs,
isolation and ledger metadata. Only one live check was used.

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

The incumbent is the newest `problems/muse_decode_layer/dev/<sN>/frozen.json` (PLAN Status). Paired whole-layer
steps, held-out six-load R [95% CI]: S18/S17 1.004321 [1.002314, 1.006332] (`5d0282bb`, prefill host, import time);
S17/S16 1.003310 [1.001006, 1.005620] (`da0cd64b`); S16/S15 1.008291 [1.005790, 1.010799] (`37c2879f`); S15/S14
1.0549 / 1.0510 / 1.0276 / 0.9895 / 1.0100 / 1.0227 at d1 / d16 / d32 / v1 / v16 / v32, geomean 1.026 (`281b1383`).
Scored: S18 1.848375x (`28b5c189`), S17 1.837887x (`5f35323a`), S16 1.824219x (`c6cc2874`), S15 1.800345x
(`9c5fc18d`), S14 1.742759x (`9b94c126`).

Reset integration decisions (2026-10-26; stage nominations require a whole-layer held-out pair):

| Recipe | Result vs control / A:A | Mechanism | Leaves open (and decision) | Receipt |
|---|---|---|---|---|
| Exact S14 final f1c77ce6 scored |1.742759x SGLang; unpaired, no A:A | Retained S14 portfolio | Retired by S15 | `9b94c126d450448a` |
| Exact S15 60254883 scored |1.800345x SGLang; unpaired, no A:A | NF projection rows + last-CTA sliding d1 + register-fold full verify | Retired by S16; paired descriptive geomean1.026 over S14 (`281b1383`) | `9c5fc18d302c4d40` |
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
| First S16 composition 2e26db54 scored |1.796154x SGLang, unpaired; no promotion CI yet | C128 d1S, DSM v1S/v16S, NORM M1, PAIR M8/16/32 on exact S15 | Superseded by nine-piece435df5f7; unpaired scores do not rank this small difference | `7139485dbeef4e0d` |
| ILP replaces PAIR in the small-M MoE | ILP/PAIR M8/16/32=1.01094/1.02063/1.01190;32 fresh seeds. A:A absolute0.488/0.377/0.364us, max1.424/0.928/1.648us | Alternate MMA accumulator chains lose to the admitted float2 epilogue despite beating S14 | Retain PAIR; close exact ILP replacement, leave combined instruction/epilogue edits open | `1b135b70835f4634` |
| Boundary/PDL archive retry | Unmeasured: LAB_ERROR combined archive exceeds lab log cap; no paired ratios or A:A available | Broad attribution archive cannot be emitted intact | No promotion or recipe closure; P7 pursues the larger MoE shell | `ce5b991a8a984a03` |
| Gated global attention repeat | C128/OLD d1S0.987744; OLD/A d16S0.9759, PP/A d32S0.9862. A:A means1.00348/1.00055/0.99918/1.00060; all4x8x16 TASK/cache gates pass | Same native source and driver as S15; OLD is the incumbent only at decode1, canonical A at decode16/32 | Corrects the old comparator scope. OLD d16S and PP d32S are new whole-layer nominees; C128 standalone timing is now measured, but its S16 off arm supersedes that selection | `bb7239afc1b3487a` |
| Cooperative verify1 sliding, recaptured control | Pooled C16/A0.952913,SE0.005071,35 seeds; A2/A3=.999262/.999499. Identical duplicate/COOP1.003740,SE.004151 passes; all7x35 TASK/cache gates pass | QT4/C16 cooperative query partition/gang; attention critical increment falls, no isolated phase attribution | Nominate exact kernel0f250847 on S15/S16; earlier failed fixed-capture controls remain unmeasured, other shapes remain open | `6a0b3f0987e94ba5` |
| Nine-piece S16 scored |1.824219x SGLang; unpaired, no A:A | PAIR/NORM, global d1S, DSM v1S/v16S, attention/MoE release, O/pool on S15 | Frozen by the separate held-out pair37c2879f; score alone is not promotion evidence | `c6cc2874ab994ee9` |
| Nine-piece S16 held-out confirmation | S16/S15 R1.008291,95% CI[1.005790,1.010799]; A:A R1.002418,seed range.999632-1.007002;16 fresh seeds/2304 TASK checks | Weighted46/16 layer sums across all six loads; entire composition wins despite d1/d16 regressions | Freeze shipped1d0c8893. DSM8/O off R.992169/.994866 selects pruning for a new pair; off arms do not veto the winning composition. Other negative point pieces also excluded from the follow-up, pending fresh confirmation | `37c2879f62554dff` |
| Pruned attention follow-up e3a919c5 scored |1.811842x SGLang; unpaired, no A:A | Drops negative off-arm points and adds OLD d16S, PP d32S and cooperative v1S | Cannot rank against S16 from the score alone; paired check c096a2c5 failed before timing | `cff1d0f93ce14a2f` |
| Pruned follow-up fresh-weight confirmation | Unmeasured: no S16_XZ85 archive, no paired R/A:A. Official host TASK passes at1.808460x | Synthetic _tune_glue preparation rejects exact expert-ID differences against its oracle, before actual TASK/timing | Neither win nor loss. Same-seed retry uses published routing margins/work checks in preparation on both arms; timed AST unchanged,144 public-oracle cases agree | `c096a2c5200447e9` |
| Guarded S15n native correctness |268/268 public TASK checks at67 shapes,2 seeds,2 graph replays; no timing/A:A measured | Guarded range dispatch9cc90d2e includes B64/B65 capacity handling | Combine with positive nearby timing a793c99e. Exact S16 merge6e43a367 has local143-fixture proof; its native combined projection-range audit is queued separately | `c03a3ae8ddfd490d` |
| TASK-semantic preparation retry scored |1.818167x SGLang; unpaired, no A:A | Same five-piece timed AST as e3a919c5; synthetic preparation validates published routing criteria and selected-expert weight/local slots | Actual host TASK passes. Same-weight/router/input-seed paired retry a8c66a6d pending; no performance promotion from this score | `93c6e7425c484608` |
| First S17 composition9c0bb03e scored |1.837887x SGLang; unpaired | Guarded S15n, P5 qt4/C16 v1S, pure large-M and VDSK release, P7 P shell; remove v16 DSM/pool128 | Paired05b8a69b is unmeasured from inherited exact-ID preparation veto; no R/CI inferred from score | `5f35323aa8bc47ef` |
| First S17 paired confirmation | Unmeasured: official host passes1.806762x; no seed timing archive | S15 _tune_glue exact-ID comparison aborts synthetic fresh-weight preparation | Neither timing win nor loss. Late revision applies TASK routing margins on every arm, all other AST unchanged | `05b8a69b98d748ad` |
| Expanded S17fea71760 scored |1.828637x SGLang; unpaired | First S17 + CHUNK4 M128/256 retaining P7 P W2/finish, fused pool/router T1/8/16, latest v1F T tail; preparation uses TASK routing semantics | [Exact composition and per-piece evidence](../problems/muse_decode_layer/dev/s17/README.md); paired da0cd64b accepted: CURRENT=ee5fe249 versus A/A2, original S16, first S17 and three leave-outs;1536+390 local TASK checks pass | `7ec5f9ab1eee4a17` |
| TASK-semantic preparation retry paired | R1.002048,95% CI[1.000129,1.003971]; A:A1.003917,CI[1.002555,1.005282];1536 TASK pass at16 unchanged fresh seeds/weights/router | New five-piece composition over prior S16; preparation uses public routing semantics on both arms, all timed AST nodes unchanged by that fix | Promote shipped4ca750a5 under declared max-SE gate. A:A mean exceeds apparent gain: marginal evidence, not a robust effect estimate. No extra small-gain sweep; larger-lever integration next | `a8c66a6d5dfc4fc2` |
| Prior S16 guarded-nearby native merge |268/268 TASK checks,67 shapes x2 seeds x2 arms; no timing or A:A measured | Exact6e43a367 merges prior nine-piece435df5f7 with guarded range dispatch | Correct native merge, but base is superseded. Current frozenee5fe249 is rebased as1468be3e and gets80-shape native validation | `e94388270a9e4c8e` |
| Current S16 guarded-nearby native merge |320/320 public TASK checks at80 shapes x2 seeds x2 arms,2 graph replays; packed candidate official host TASK passes, check host1.813381x. No new timing/A:A claim | Whole guarded range lookup on exact currentee5fe249, including13 official cases and B64/B65 | Freeze shippedd2b83953/expanded1468be3e under explicit nearby criterion froma793c99e: nearby gaps improve, scored dispatch unchanged. Exact final is not yet scored; larger-lever composition next | `e26bc28cfe8449f5` |

Original S15 admission wrapper49ce378b was rejected422 before a job existed: its packed source had no
explicit top-level `kernel_fn`. Entry-only repair2a4d0dde passed normal/forced full-wrapper screens and ran
as281b1383. The old per-load/host-equivalence veto was superseded by the score policy in97e98f3;
exact60254883 is owner-frozen. That legacy receipt lacks raw seed covariance and per-arm TASK checks,
so its geomean is descriptive rather than a new held-out confidence claim. S16 records both explicitly.

Admission history through S15 (later S16 composition/off-arm receipts above supersede these selections; r > 1 means removing the piece is slower):

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

### S19 timeline and lever re-rank

**Measured:** `9c16c850f7b941a5`, one prefill-host import lab, exact S19 `62ea5bf4` / shipment
`8b415553`; all12 sliding/full shapes,3 prospective fresh seeds x3 cold CUPTI rounds. Both untouched
and mapped arms pass72 public TASK checks; full kernel/resource/stream/DAG fingerprints match,
monotone timestamps and exact critical accounting (max error0). Source/observer/checker/payload pins
and packet checksum verified. The outer host's deliberate prepare stop is not an experiment failure.
All108 profile rounds and288 plain/mapped event controls remain at integer-ns precision; no trimming.
[Per-stage/load and all12 shape tables](../problems/muse_decode_layer/dev/s19_timeline/result/H200.md),
[rounds](../problems/muse_decode_layer/dev/s19_timeline/result/rounds.csv.xz),
[validation](../problems/muse_decode_layer/dev/s19_timeline/result/validation.json),
[priors/ranking](../problems/muse_decode_layer/dev/s19_timeline/result/ranking.json),
[reproduction](../problems/muse_decode_layer/dev/s19_timeline/README.md).

Unlike the S15 fitted-stream model, requested roofs here are max(actual lossless-layout read+write
bytes/4.8TB/s, tensor FLOPs/989TFLOP/s), with no intercept. S19's raw P10 pool bank is counted as raw;
selected NF/CD weights retain their coded storage. Excess is positive critical time above1.3roof,
averaged over every round. Busy DAG gaps go to actual overlapping work, never launch overhead;
concurrent experts share one unique-work envelope. All small-M W13/W2/finish work is fused and stays
one measured row; large-M W13/W2/finalize remain separate. Scalar reductions/softmax/selection are
not modeled by byte/tensor roofs. Logical bytes are not HBM counters. The shape-specific observer
mapped/plain range0.98797–1.00176 and profiled/mapped0.99839–1.01253 are timing controls, not a gain.

**Modeled:** S19 score1.875109x = S18 board1.848375 x paired S19/S18 lab1.014463; scored S19 is pending.
Close target2.6 means D=ln(2.6/1.875109)=.326845, replacing the older ln1.284 denominator. Stage-family
ideal-roof shares are small-M MoE41.1%, large-M27.0%, dense25.7% (37.1% with ideal exact12), native
sliding13.1%, full/FA3 including actual v16S16.8%. Their1.3roof target shares are21.4/14.3/11.1
(20.9% ideal exact12)/12.7/10.4%. All-stage1.3roof scenario R1.36236 removes94.6%D; counterfactual
ceilings overlap and are not additive. This fresh routing cohort is not a paired S19/S15 speedup test.

**Uncovered:** b1/b2 normalization/reduction and route/metadata compute expose6.79–7.26us per weighted
layer above1.3roof,13.0%D (14.1% ideal-roof ceiling). P7's prefetch and P11's materialization do not
cover that math; PLAN now names an unassigned lever. Separate MoE finalize stays within P3b.
S19 ramp+idle R1.08543 is25.1%D, a scenario, not a physical continuity bound; materialization alone
R1.01470 is4.5%D, plus true idle R1.01949 is5.9%D. No fresh synchronization-only measurement supports
carrying the old S15 persistent-kernel estimate forward; issue/reuse beyond byte retention is unmeasured.

Today's mechanisms change odds independently of those family caps: ca022370's parked consumer deletion
R1.01590 covers4.8%D, and554c392d's wait-only R1.01667 models5.1%D. New matched ladder da1fb2ee has
conditional decode-scale/MMA+activation contrasts R1.00718/1.04564 (~2.2/13.7%D); sinks/transport and
the slower STREAM control prevent calling them exclusive pipe costs or a family closure. P10's newer
e4030204 split-only deletion R1.002444 covers0.75%D, versus producer deletion1.086449 (25.4%D) and
whole-consumer deletion1.019571 (5.9%D); c87c2d8a's24 native dispatch arms yield no nominees. Its prior
7f110a37 large_split builds still have no winner. P5b's9dc1159a ~.89 FA3-call nominees are not an11%
whole-layer gain; R-20261026-1's public ports now have native receipt26d5054f:704TASK,8preferred cells; L v1F/v16S .757719/.763592 and O v1S .780899, L v32S .750479. These are component ratios; P4's held-out whole-layer G0 remains pending.

Judgment priors and forecast hours make ceiling*p/cost explicit: P5b, P5, P3a, P10, unassigned glue/route,
P3b, P7, P11. Completed26d5054f raises attention odds; it does not change the parked consumer-family decision. P4 stays first in execution for the live integration/scored path. Only existing row
order/rank numbers and Levers changed; lane-owned work text, states and pending runs are preserved.

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
which is also included as a control. Audited H200 ratios are below; the first wrapper lost its archive and the bounded retry completed.

| Recipe | Ratio / measured prerequisite | Mechanism / decision | Leaves open | Receipt / evidence |
|---|---|---|---|---|
| BLOCK | Premise confirmed; expert-wide `NT1*nblk[e]` wait confirmed in S15 | W2 acquires `done[e,nb]` againstNT1;128 done slots/512B at M32. W13 store→fence→barrier→publish→acquire preserved. Only M16/M32 | Actual wait reduction and counter-address cost on H200 | `review/claims.json`, `review/manifest.json` |
| CANCEL | Premise confirmed; W2 `v*=256*gs2[e]` before post-norm/RMS confirmed | Drop that element multiply,adjust eps/a²,retain sign,guard zero scale/norm and handle epsilon range in double. W13/SwiGLU unchanged | H200 savings versus finalizer/metadata cost and six-load score | `review/extreme_validation.json`:160 helper cases; `component_validation.json`:94 TASK checks |
| SHARD2 | Premise confirmed; PAIR's single norm slot confirmed |2 atomic slots per pair,CTA-indexed; W13 zeros both,finalizer sums2 | Contention relief versus extra zero/load work; no LOCAL export | `review/component_validation.json`,exact patch |
| SHARD4 | Premise confirmed; same PAIR parent |4 independent norm slots per pair | Same trade-off at4 shards; interactions with block readiness | `review/component_validation.json`,exact patch |
| SHARD8 | Premise confirmed; same PAIR parent |8 independent norm slots per pair | Same trade-off at8 shards; other reduction ownership | `review/component_validation.json`,exact patch |
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

Review job**`1da2153a0f5543b9`** passed its official host, but the wrapper emitted
`LAB_TRUNCATED 2 middle lines`, discarding GATE and the full timing archive. **Every arm is unmeasured**;
this closes no recipe and nominates nothing. The local6,399-character compressed archive did not bound
H200 clock entropy. `review/receipt_truncation.json` preserves the receipt and cause.
The fixed raw transport retains every per-seed1ppm ratio and every independent1ns A:A pair; it omits
candidate absolute times/CUPTI nodes from transport. Worst-case output8,845characters, independent of
entropy. Exact packed local payload passes16seeds/64AA at7,726characters/49.4s. Kernels are byte-identical
to all14 successful screens. Retry check**`b57d691d689046a8`**,sender1365722,wrapper`3457baa10244c861`,
completed on MoE; audited per-arm results are below. The first1,432,371-byte source rejection remains separate evidence.

### P3a S15 review receipt and composition (2026-10-26)

Original `1da2153a0f5543b9` remains unmeasured: its wrapper discarded both correctness and timing archive.
Retry **`b57d691d689046a8`** is complete:16 fresh seeds per M1/8/16/32; all official/native TASK gates pass,
no arm failures, worst total/row .002401/.002545. Candidate/control time ratios below; SE in parentheses.
A:A mean absolute us **.200/.541/.409/.391**, maxima .736/1.280/1.312/.896, ordered M1/8/16/32.
A:A mean ratios1.000615/1.002764/1.000122/1.000417; descriptive, never a per-shape veto.

| Recipe | Ratio vs S15 at M1 / 8 / 16 / 32 (SE),n16 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| BLOCK | 0.99371/1.00218/0.99237/0.99406 (0.00168/0.00129/0.00208/0.00292) | Own-token-block release/acquire readiness; store/fence/barrier/publish order preserved. Nominate M16/M32; M1/M8 are exact S15 control aliases. | PAIR+BLOCK interaction and whole-layer held-out R; other dependency schedules. | `b57d691d689046a8` |
| CANCEL | 1.00167/1.02266/1.01870/1.01407 (0.00182/0.00210/0.00144/0.00107) | W2 scale removed with sign,eps/a²,zero/extreme handling; tested implementation is slower at every M. No nomination. | Cheaper metadata/finalizer realization of this valid identity; W13 is not cancellable through SwiGLU. | `b57d691d689046a8` |
| PAIR | 1.00192/0.98090/0.97231/0.97999 (0.00168/0.00239/0.00291/0.00283) | Two writer warps and contiguous pair planes; nominate M8/M16/M32. New M1 estimate loses; earlier positive M1 receipt remains evidence. | Block readiness composition, M1 choice versus NORM/shards, and held-out whole-layer R. | `b57d691d689046a8` |
| SHARD2 | 0.99646/0.98603/0.97600/0.98517 (0.00126/0.00299/0.00390/0.00341); /PAIR 0.99179/1.00362/1.00351/1.00674 | PAIR plus2 norm slots; nominate all4 M versus S15 and M1 versus PAIR. At M8+ it loses directly to PAIR. | M1 held-out gain; other shard ownership or layouts. This closes only2-slot choice relative to PAIR at M8+. | `b57d691d689046a8` |
| SHARD4 | 1.00410/0.98385/0.97798/0.98561 (0.00226/0.00348/0.00353/0.00327); /PAIR 0.99911/1.00073/1.00716/1.00332 | PAIR plus4 slots; nominate M8/M16/M32 versus S15 and M1 versus PAIR (M1/S15 loses). Direct M8+ loses. | Per-shape whole-layer choice; different reduction ownership. This does not close all sharding. | `b57d691d689046a8` |
| SHARD8 | 0.99876/0.98759/0.98019/0.98910 (0.00162/0.00345/0.00292/0.00306); /PAIR 0.99646/1.00698/1.00636/1.01027 | PAIR plus8 slots; nominate all4 M versus S15 and M1 versus PAIR. Direct M8+ loses. | Held-out M1 alternative; different placement/ownership and six-load score. | `b57d691d689046a8` |
| PACK5 | 1.28466/1.39238/1.40819/1.45243 (0.00788/0.00895/0.00554/0.00492) | Lossless scale packets cut net bytes4.058%; timed unpack/metadata costs outweigh the byte saving,1.28–1.45x. Close this exact format/decoder. | Lower-issue packet layouts/decoders and checkpoint distributions; lossless scale compression itself remains open. | `b57d691d689046a8` |

`point_nominations.json` now includes all19 implementation point-win comparisons plus the explicitly marked
BLOCK M1 control alias (handoff=false; no changed implementation at M1), with SE,16 seed indices,weight/router seed recipes and exact S15 patches/hashes.
SHARD4 M1 is included for its direct PAIR point win despite its S15 loss. The original independent kernels
are unchanged by the receipt repair. `review/results.json` retains every seed ratio and A:A pair.

Next: one per-shape composition arm, **SHARD2 at M1, PAIR at M8, PAIR+BLOCK at M16/M32**, compared
with S15,PAIR and the best independent choice on fresh weight seeds275353+7919*s and routing279000+M+97*s.
CANCEL/PACK5 are excluded by their measured losses. No additive or whole-layer gain is claimed;
P4 composes and confirms held-out six-load R. Exact submitted composition: `review/combo/combo.patch`;measured receipt follows.

Composition admission:270 native component/replay checks; normal/forced each pass13 TASK cases,
12 dispatch shapes,28 static checks and eager/captured calls for every variant. Exact packed lab
passes16 fresh seeds/64AA,4,371characters/79.9s locally; no local timing claim. Sent as MoE check,
sender**1548101**,wrapper**`da4bfcfab9f0b0db`**,priority1.2/import budget240s; accepted judge check**`c49dbdffe92d491a`**,receipt complete (audited below).
Exact patch `review/combo/combo.patch`,candidate6884c929; admission and queue records under `review/combo/`.

### P3a held-out composition receipt and next instruction cuts (2026-10-26)

Check **`c49dbdffe92d491a`** is complete:16 fresh seeds per M1/8/16/32,256 paired comparisons,64 A:A pairs;
all native/official TASK gates pass,no failures,worst total/row .002419/.002520. Exact source audit and seed matrix:
`review/combo/results.json`. A:A mean absolute .225/.480/.309/.362us,max.688/1.552/1.136/.912.
These spreads describe uncertainty; the stage point gate nominates all4 M versus S15. Only P4 confirms whole-layer R.

| Recipe | Ratio vs S15 (SE),n16; /BEST (SE) | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| SHARD2 M1 | 0.997261 (0.001758); /BEST 1.001877 (0.001619) | Fresh-seed point win; CUDA matches the independent control,so direct difference is control noise. Nominate to P4. | Whole-layer held-out R,nearby shapes,and other per-shape alternatives. | `c49dbdffe92d491a` |
| PAIR M8 | 0.988757 (0.002226); /BEST 1.001757 (0.001838) | Fresh-seed point win; CUDA matches the independent control,so direct difference is control noise. Nominate to P4. | Whole-layer held-out R,nearby shapes,and other per-shape alternatives. | `c49dbdffe92d491a` |
| PAIR+BLOCK M16 | 0.970993 (0.002883); /BEST 0.995504 (0.002804) | Own-block W2 readiness composes with PAIR; point win also versus the direct PAIR/BEST control. Nominate to P4. | Whole-layer held-out R,nearby shapes,and other per-shape alternatives. | `c49dbdffe92d491a` |
| PAIR+BLOCK M32 | 0.979561 (0.003839); /BEST 0.993125 (0.002482) | Own-block W2 readiness composes with PAIR; point win also versus the direct PAIR/BEST control. Nominate to P4. | Whole-layer held-out R,nearby shapes,and other per-shape alternatives. | `c49dbdffe92d491a` |

P4 exact combined patch `review/combo/combo.patch`,SHA256`6b3990b2e7bdcf0f4862df49f2e5ce0bfe882a1b1d862fd973a6f9582bb38b6a`,
onto S15`60254883`,candidate`6884c929`. `point_nominations.json` includes every eligible COMBO/PAIR/BEST point comparison,
SE,seed indices/recipes and exact patch; M16/M32 direct ratios .995504/.993125 also nominate. No per-shape confidence veto.

Next arms use this measured per-M parent. Count builds use sm90a,-O3,-lineinfo; complete SASS opcode/resource evidence
is `review/next_issue/sass_cost.json`. Whole static instruction counts include exceptional paths; they are not dynamic issue/stall counts.

| Recipe / audit | Ratio / prerequisite | Mechanism / decision | Leaves open | Receipt / evidence |
|---|---|---|---|---|
| Original CANCEL cost | Measured /S15 1.00167/1.02266/1.01870/1.01407; flags69 whole static instructions3168→4072,text50688→65152B | Added expert gathers,metadata loads,sign/range predicates and duplicated FP64 tails:LDG77→100,LDS143→159,FSETP15→76,64DMUL+169DFMA. Both125regs,zero spills: there is no added register count or spilling to explain the loss. Ordinary finite epsilon takes the FP32 path; FP64 text alone does not prove normal FP64 execution. | Dynamic attribution among the extra common-path work and text footprint. | `b57d691d689046a8`;`next_issue/sass_cost.json` |
| FAST cancellation | H200 unmeasured; helper224 cases,worst9.33e-8 relative;16 full-MoE scale TASK checks pass | Same identity with bounded FP32 metadata(q=a/sqrt(eps),mul=min(1,q²),bias=4096*min(1,1/q²),num=64*sign(q)*min(1,abs(q))). Factor=num*rsqrt(fma(sum,mul,bias)),zero norm guarded. No runtime FP64 or sign/range branch. FAST/S15 static3264 instructions,84LDG,23FSETP; PAIR+BLOCK FAST uses124regs,zero spills. | Whether fewer instructions beat the remaining expert gathers/vector-metadata cost on H200. | `next_issue/fast_extremes.json`,`native_audit.json` |
| INDEX B-fill | H200 unmeasured; marked W13 integer/address/control31→13,W2 26→8;16 vector loads retained each | Hoist row-valid/pass branches outside8 fills; one bounded32bit vector index per segment. Save18/(16*L)=1.125/L instructions per16 requested weights,where L is actual segment tiles. Core decoder unchanged; full static counts are not a per-tile count. PAIR+BLOCK registers125→122,zero spills. | H200 benefit after segment-length amortization; reductions in the per-tile decoder itself. | `next_issue/sass_cost.json`,marked-fill probes |
| FLUSH1 | H200 unmeasured; two consecutive calls confirmed after first resetsnv0=0 | Delete only the second zero norm flush at a segment transition; retain every required publication/fence and final flush. Count build PAIR+BLOCK static3352→3296,ATOMG7→6;125regs,zero spills. | H200 benefit from less atomic/control work; other reduction ownership. | `next_issue/build.py`,`sass_cost.json` |

All9 new cell variants pass90 native TASK/replay checks each,810 total (including empty/all8 routes,
M9/M24,nonzero expert starts and changed captured inputs). The three arms remain separate so the effect is readable.
Normal/forced screens and packed H200 follow-up are recorded under `review/next_issue/`.

Next paired check accepted as **`d7fb2d8bddae4b8c`**,sender**1971066**,wrapper**`76a7f192a080d0b6`**,
MoE,priority1.2/import budget240s. FAST/INDEX/FLUSH1 are separate arms versus the measured c49 BASE and S15,
16 fresh seeds/M1/8/16/32,448 paired ratios,64 A:A pairs,TASK before/after. All6 normal/forced screens pass
13 TASK cases,12 shapes,28 static checks each;810 native component/replay checks,224 helper extreme cases
and16 captured full-MoE scale tests pass. Exact packed local lab completes16seeds/64AA,5,919characters/94.7s;
local timing is not H200 evidence. Entropy-independent receipt bound7,288characters;wrapper975,473B.
`review/next_issue/admission.json` and `queue.json` pin the exact admitted source. All new H200 ratios remain
measured below; static instruction reduction alone did not establish a gain. Dynamic attribution,
segment-length amortization,other decoder layouts,and P4's whole-layer R remain open.

### P3a instruction-cut verdict and structural lever (2026-10-26)

**`d7fb2d8bddae4b8c`** completes16 seeds/M,448 paired comparisons and64 A:A pairs;all official/native TASK gates pass,
no arm failures,worst total/row .002431/.002531. Controls are exact c49 per-M BASE plus S15; ratios are candidate/control time.
A:A mean absolute .183/.606/.438/.252us,max.384/1.488/1.424/.720 at M1/8/16/32. Intervals below are
descriptive95% Student-t across16 seed ratios (t15=2.13145);only P4 confirms held-out whole-layer R.

| Recipe | Ratio / BASE [95% CI] at M1;M8;M16;M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| FAST | 0.997548 [0.992767,1.002329]; 1.006293 [1.001147,1.011439]; 1.004684 [1.001688,1.007681]; 1.003011 [1.000561,1.005462]; /S15 0.998218/0.995274/0.973282/0.979448 | Bounded FP32 scale-cancellation metadata removed runtime sign/range/FP64 paths. M1 point win;M8+ lose directly. All4 M point-win versus S15 as alternatives. Every valid point win handed P4 with exact patch,SE,seeds and CI. | Other finalizer/gather layouts;the identity remains valid. No standalone sub1% confirmation run. | `d7fb2d8bddae4b8c` |
| INDEX | 0.992518 [0.987187,0.997849]; 1.002460 [0.995847,1.009072]; 1.009998 [1.005204,1.014793]; 1.012458 [1.010286,1.014630]; /S15 0.991563/0.986110/0.980997/0.989037 | Hoisted B-fill branches and bounded32bit vector indices cut18 integer/control ops per segment. M1 wins;M8+ lose despite fewer instructions and122regs. All4 M win versus S15 as alternatives. Every valid point win handed P4 with exact patch,SE,seeds and CI. | Other schedules and per-tile decode cuts;static count reduction does not establish latency gain. | `d7fb2d8bddae4b8c` |
| FLUSH1 | 1.003517 [0.999814,1.007219]; 1.000782 [0.994676,1.006888]; 1.000861 [0.996048,1.005674]; 1.000898 [0.998790,1.003007]; /S15 1.004497/0.986165/0.971168/0.977198 | Removed the duplicate zero norm flush/atomic. No point win versus BASE at any M;M8+ remain alternatives versus S15 through inherited PAIR gains. Every valid point win handed P4 with exact patch,SE,seeds and CI. | Other norm ownership;close only this removal as an incremental recipe. | `d7fb2d8bddae4b8c` |

P4 handoff `point_nominations.json` includes all13 eligible FAST/INDEX/FLUSH1 comparisons;the direct S15/BASE M1
control point difference is retained in the receipt,not a new kernel patch. No additional small-cut-only run is sent.

Structural sizing reads the full oracle receipts,not just ratios: `360810b110a44d36` loads-only C removes readiness,
activation/MMA and finalizer as well as dequant,and retains depth2. Its .74514/.72301/.66836 ratios therefore overlap
with the dequant/readiness costs and do not isolate depth. `c768a8e361f54977` critical M32 CTA has W2-ready11.58us
and grid barrier4.98us within75.98us;these are one diagnostic critical-path sample,not guaranteed additive savings.
The structural depth/phase batch below tests these hypotheses directly;oracle savings are not added.

### P3a depth and phase-boundary verdict (2026-10-26)

**`48dc8b04902b40b1`** completes16 fresh seeds at M8/16/32,528 paired ratios and48 A:A pairs. All official/native TASK gates pass;no failed arms,worst total/row .002378/.002526. Ratios are candidate/control time;95% CI is descriptive Student-t15 across seed pairs. A:A mean absolute .525/.270/.334us,max1.280/.896/.800. BASE is exact c49 PAIR M8/PAIR+BLOCK M16/32.

| Recipe | /BASE [95% CI], M8;M16;M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| D3 | 1.075060 [1.067191,1.082929]; 1.100324 [1.085194,1.115454]; 1.107974 [1.093716,1.122232] | Three raw TMA slots;2→1resident CTA. Slower at all M. | Other layouts that deepen lookahead without halving residency. | `48dc8b04902b40b1` |
| D4 | 1.086272 [1.077668,1.094876]; 1.110162 [1.095577,1.124748]; 1.115429 [1.101645,1.129213] | Four raw TMA slots;2→1resident CTA. Slower at all M. | More lookahead with smaller per-slot footprint or producer partitioning. | `48dc8b04902b40b1` |
| SPLIT2 | 1.029222 [1.015666,1.042779]; 1.029733 [1.014861,1.044606]; 1.001459 [0.988354,1.014563] | Ordered W13/W2/combine launches remove W13 publication/W2 acquire and final grid spin;pay extra launches and repeated routes. No point win versus BASE;M32 /S15 .976908 [ .968646,.985169 ] is a P4 alternative. | A fused dependency schedule retaining2CTA residency and route reuse;whole-layer R for the M32 S15 alternative. | `48dc8b04902b40b1` |
| SPLIT3 | 1.151537 [1.140801,1.162273]; 1.158436 [1.144424,1.172447]; 1.147482 [1.133501,1.161464] | Same spin removal with3 raw slots;1resident CTA. Slower at all M. | Dependency removal with smaller shared footprint and fewer route rebuilds. | `48dc8b04902b40b1` |
| SPLIT4 | 1.176156 [1.163619,1.188694]; 1.180693 [1.166277,1.195108]; 1.161173 [1.146393,1.175954] | Same spin removal with4 raw slots;1resident CTA. Slower at all M. | A schedule that removes waits while retaining residency and one route construction. | `48dc8b04902b40b1` |

H200 resources confirm fuseddepth2=2residentCTAs,125regs,84.2–85.1KiB dynamic shared;depth3/4=1CTA,121.1–122.0/157.9–158.9KiB. Split math124regs,combine40regs;zero local/spills. These measured losses close the exact residency/ordered-launch recipes,not the wait/depth lever. SPLIT2 M32/S15 point estimate and exact patch are in `dev/nt1_timeline/point_nominations.json`;P4 confirms whole-layer R. Next structural step is dequant issue with a lane-replicated exact byte-pair table,depth1/2 in one batch.

### P3a structural dequant verdict (2026-10-26)

**`11b3849eb86749df`** completes16 fresh seeds at M8/16/32,240 paired comparisons and48 A:A pairs;all official/native TASK gates pass,no failed arms,worst total/row .002377/.002512. Exact submitted payload/cells/selectors and code/scale preservation proofs are audited. Candidate/control time and descriptive95% Student-t15 CI below. A:A mean absolute .569/.344/.390us,max1.072/1.056/.752.

| Recipe | /BASE [95% CI], M8;M16;M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| LUT1 | 1.185044 [1.174661,1.195427]; 1.145672 [1.122240,1.169103]; 1.123305 [1.113185,1.133425] | Fixed64KiB lane-bank-replicated exact code-pair table replaces15 core SHF/LOP3 per16weights with8PRMT+8LDS;scale HMUL2 retained. One raw stage,2resident CTAs,127regs/zero local. The net initialization/shared/depth trade-off loses;individual costs are not separately timed. Loses every M versus BASE andS15;no P4 nomination. | Other direct packed decoders or smaller tables that retain lookahead;this does not close dequant issue. No dynamic pipe attribution is inferred from static counts. | `11b3849eb86749df` |
| LUT2 | 1.174630 [1.161741,1.187520]; 1.190589 [1.169105,1.212072]; 1.222294 [1.208673,1.235915] | Same exact decoder with2 raw stages;64KiB static table lowers residency2→1. All math/codes/scales preserved;loses every M versus BASE andS15. | A decoder that removes integer work without this shared footprint/initialization cost;different overlap/scheduling. | `11b3849eb86749df` |

H200 executes the actual64KiB table decoder and both raw depths;local sm120 full-kernel stand-in was explicitly32KiB/one stage,with separate32768-case native64KiB address proof. All4 normal/forced whole-layer screens pass13TASK/12shapes/27static each;116 native current-input/capture TASK checks and1.61B-code/raw-scale bitproof pass. Full-image static SHF+LOP3 changes409→181–184,PRMT/LDS add128sites each;counts are not measured stalls. Both layouts lose,including LUT1 with2CTA residency;residency alone is not a sufficient explanation.

P4 S18 handoff `dev/nt1_timeline/point_nominations.json` retains all eligible d7 FAST/INDEX/FLUSH1 alternatives and48dc SPLIT2 M32/S15;no decoder point winner. Next largest open structural test:reclaim dead route/reduction scratch so extra raw lookahead can retain2CTA residency,first by exact code/lifetime/resource audit. Fused dependency ownership remains open after ordered-launch costs erased the spin savings.

### P3a three full slots at two CTAs (2026-10-26)

**`cdd07707a6974cac`**,sender3240664,wrapper5daf36f333fec1e5 completes16fresh seeds at M8/16/32,624paired ratios48A:A. All official/native TASK gates pass,no failed arms,worst total/row .002372/.002532. Candidate/control time below;95% CI is descriptive Student-t15 across seed pairs. A:A mean absolute .487/.275/.357us,max1.904/.848/1.120. BASE is exact c49 PAIR M8/PAIR+BLOCK M16/32.

Cheapest local sizing used exact sm90a ptxas plus CUDA13 cuda_occupancy.h:233472B shared/SM,1024B driver reservation/block,128B allocation granularity,65536registers/SM,256threads. Two CTAs allow115712 useful shared bytes and128registers/thread after allocation rounding. Every18H200 resource row matches the offline count;actual local bytes0 throughout.

| Layout | Dynamic shared bytes,M8/M16/M32 | Registers,M8/M16/M32 | Actual CTAs |
|---|---|---|---|
| BASE2 |84176/84496/85136|125/125/125|2|
| SINGLE2 /SINGLE3 |78608/78672/78800 /115488/115552/115680|125/121/121|2/2|
| HALF2 /HALF3 |77968/78224/78736 /114848/115104/115616|122/122/122|2/2|
| DOUBLE2 /DOUBLE3 |78608/78672/78800 /115488/115552/115680|125/125/125|2/2|

| Recipe | /BASE [95% CI],M8;M16;M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| SINGLE2 | 1.030108 [1.023816,1.036401]; 1.025072 [1.020376,1.029768]; 1.033352 [1.029474,1.037230] | One4KiB reduction buffer;route scratch aliases;exact16bit offsets/8bit reverse indices;inverse route rebuilt after math. Extra reuse and reconstruction barriers. | Other scratch schedules that preserve the old overlap without rebuilding route. | `cdd07707a6974cac` |
| SINGLE3 | 1.058302 [1.050464,1.066139]; 1.066341 [1.062101,1.070581]; 1.057700 [1.054786,1.060615] | Same SINGLE layout with3fullrawslots;actually2CTAs,125/121regs,zero local. Loses versus BASE andS15. | Producer issue independent of consumer barriers;L2 lookahead without extra shared footprint;L1/shared carveout contribution is unmeasured. | `cdd07707a6974cac` |
| HALF2 | 1.046150 [1.039883,1.052417]; 1.037873 [1.031985,1.043760]; 1.042459 [1.039524,1.045395] | One2KiB paired-warp reduction,expert-byte scratch aliases. Extra store/add/reuse barriers;FP32 sum order changes legally. | Other reduction mappings without these barriers;no precision or work reduction. | `cdd07707a6974cac` |
| HALF3 | 1.067235 [1.061502,1.072969]; 1.078475 [1.073110,1.083839]; 1.071193 [1.069244,1.073142] | Same HALF layout with3fullrawslots;actually2CTAs,122regs,zero local. Loses versus BASE andS15. | Asynchronous lookahead without this shared/L1 footprint and extra reduction synchronization. | `cdd07707a6974cac` |
| DOUBLE2 | 1.010738 [1.005133,1.016343]; 1.008580 [1.004133,1.013027]; 1.008283 [1.005613,1.010954] | Two2KiB paired-warp reduction buffers plus exact compact/rebuilt route. /BASE loses; /S15 .983936/.981001/.981782 point alternatives to P4,not incremental wins over PAIR/BLOCK. | P4 whole-layer R for the alternative;producer/transport changes that preserve original reduction overlap. | `cdd07707a6974cac` |
| DOUBLE3 | 1.026589 [1.018336,1.034841]; 1.044325 [1.039645,1.049005]; 1.035255 [1.032020,1.038491] | Same DOUBLE layout with3fullrawslots;actually2CTAs,125regs,zero local. Loses versus BASE andS15. | Timed warp-cooperative L2 lookahead at2rawslots or a separate producer;this closes only the tested3slot/scratch schedule. | `cdd07707a6974cac` |

This changes the residency diagnosis:keeping2CTAs does not make these deeper-lookahead schedules win. All6arms lose to BASE;all3depth3arms also lose to S15. BASE-normalized depth3/depth2 contrasts are SINGLE1.02747/1.04036/1.02363,HALF1.02021/1.03920/1.02758,DOUBLE1.01587/1.03549/1.02677. These are quotients of separately BASE-paired seed ratios,not direct depth3/depth2 timing pairs;`depth_contrast.json` retains SE/CI/seeds. No inference isolates L1/shared carveout,producer issue or barrier time. The360810b1 loads-only oracle also removes math/readiness/finalizer and does not prove a depth-only shortage.

All6selectors pass348native TASK/current-input/capture checks,includingM9/M24,all8localroutes,nonzero local starts andchanged replay inputs. All6normal/forced deep-layout whole-layer screens pass13TASK/12shapes/27static each;two-slot controls share exact CUDA andget native selector/capture checks. Explicit local3→2slot stand-in is absent from the H200 payload;H200 runs actual3slots. Fourfullrawslots alone need147456B>115712B,so no4slot arm was sent.

P4 S18 receives DOUBLE2 at M8/16/32 versusS15,all16seed ratios/SE/CI andexact `structural/occupancy/double2.patch` SHA2564cfcbab352a334d254b98ab01be9777e5d68a0fa720d8e826fd06e59abbae795 in `point_nominations.json`. No win versus strong BASE andno3slot nomination. Next structural hypothesis:timed warp-cooperative L2 lookahead before raw-copy waits,retaining2rawslots to avoid extra shared footprint;first count/audit producer issue andsynchronization. No standalone sub1% confirmation run.

### P3a dedicated load issue and compute overlap (2026-10-26)

**`20ef2355a317497f`**, sender3928502, wrapper3dbe0bf239cabd95, completes16 fresh seeds at M8/16/32,432 paired ratios and48 A:A. All official/native TASK gates pass, no failed arms; worst total/row .002372/.002507. Exact submitted payload/cells/selectors, checksummed packet and all12 runtime resource rows are audited. Candidate/control time; descriptive95% Student-t15 CI below. A:A mean ratio1.001648/1.001493/.999181, SE .001714/.001389/.000775; mean absolute .324/.375/.198us, max1.008/1.152/.800. BASE is c49 PAIR M8 andPAIR+BLOCK M16/32; M1 is unchanged.

| Recipe | /BASE [95% CI], M8; M16; M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| CACHE | 1.066665 [1.062034, 1.071295]; 1.073981 [1.066306, 1.081655]; 1.080346 [1.076033, 1.084659] | Moves half the BF16 activation operands from registers to 32 KiB shared; aliases route scratch and rebuilds inverse routes. Coupled warp7 load issue retained. | Other activation/register layouts without this shared and route-rebuild cost. | `20ef2355a317497f` |
| WS4 | 1.009828 [1.002932, 1.016723]; 1.017186 [1.009144, 1.025229]; 1.018867 [1.009796, 1.027938] | Four consumer warps, one active producer in a four-warp producer group; setmaxnreg 128 initial to 192/32. Removes load issue from math warps, doubles K work per consumer, retains B operands in registers. Loses to BASE; beats S15 at all three M. | P4 whole-layer R for the exact alternative; other producer/consumer partitions that retain eight math warps without shared B-cache. | `20ef2355a317497f` |
| WS8 | 1.060350 [1.053172, 1.067528]; 1.070122 [1.062354, 1.077891]; 1.076988 [1.072402, 1.081575] | Eight consumers plus producer group; setmaxnreg 80 initial to 104/32, same shared activation/route layout as CACHE. Producer executes every TMA, no decode/MMA. Net schedule loses. | Register-efficient eight-consumer schedules without the B-cache/rebuilt-route cost; clock evidence does not isolate exclusive issue stalls. | `20ef2355a317497f` |
| WS8L2 | 1.127164 [1.109023, 1.145305]; 1.115681 [1.097802, 1.133560]; 1.095072 [1.088500, 1.101644] | WS8 plus one bulk L2 prefetch of its next own-stream MoE weight tile. No cross-stage prefetch or persistence. Loses at every M. | Other demand-copy schedules; this does not close all within-stream lookahead forms, but provides no reason to repeat this form. | `20ef2355a317497f` |

Per-warp static issue model is pinned to the PIPE `16c180784f414576` receipt: current BASE decode/MMA body is byte-identical to measured S14. Per256 weights/lane:416 decode/scale instructions including128 HMUL2,32 HMMA,9 weight/scale LDS,457 total. Segment B-fill adds16 vector LDG and~46 instructions; PAIR epilogue models148 W13/129 W2 warp instructions per CTA tile, mostly warps0–1. BASE warp7 does all math and issues the next TMA after the consumer partial-store barrier. This establishes source coupling; same-op rate envelopes do not establish hardware issue utilization or dequant stalls.

The new four-seed clocks retain every active warp in `structural/overlap/clock_results.json`: median across seeds of per-active-CTA mean warp residence,32ns precision. Selected rows:

| M | BASE warp7: dequant/MMA; copy address/issue; slot wait (us) | BASE warp0: B-fill; ready; epilogue (us) | WS8 warp8 producer: copy; empty-slot wait (us) | Clock observer BASE / WS8 |
|---|---|---|---|---|
|8|22.816; 6.976; 2.816|1.344; 3.328; 6.432|11.232; 52.000|1.126 / 1.340|
|16|25.536; 7.840; 3.008|2.624; 4.032; 7.296|13.024; 56.064|1.144 / 1.307|
|32|34.144; 10.976; 3.968|3.264; 5.056; 9.760|17.664; 74.272|1.137 / 1.263|

BASE warp0 issues the initial2 copies; warp7 issues20/23/31, with22/25/33 average math tiles per warp. WS8's warp8 issues22/25/33 copies and performs zero math; all8 consumers issue zero copies. Separate producer implementation is proven, and all performance arms keep2 actual CTAs withzero local bytes. Runtime registers BASE/CACHE/WS4/WS8/WS8L2 are125/85/128/80/80; dynamic shared budgets in the lane README. Clock probes allocate8B BASE/24B WS8 local bytes (static spill stores4B/24B), still2CTAs. Clock observer ratios above (SE BASE .0082/.0039/.0050, WS8 .0049/.0102/.0047) forbid treating phase differences as unbiased normal-kernel costs. The first8 issue endpoint samples of the seed0 critical CTA fall inside warp0 math in0/8 BASE events at all M and0/8,0/8,1/8 WS8 events; issue endpoints cannot establish where the entire issue interval lies. No exclusive stall or additive oracle attribution is claimed.

WS4/S15 M8/M16/M32 is 0.991669 [0.984621, 0.998718]; 0.989111 [0.982694, 0.995529]; 0.991728 [0.986459, 0.996996]. SE .003307/.003011/.002472, 16 seeds each. `point_nominations.json` hands P4 all three alternatives with exact `structural/overlap/ws4.patch`, SHA256 `4a4ad0258e2eac58d1b2543bda65c39be9ed4cce4a37f45122296c7ce29cd51c`. These are not incremental wins over the PAIR/BLOCK incumbent; only P4 held-out six-load R can promote.

All4 arms pass232 native TASK/current-input/capture checks and8 normal/forced whole-layer screens (13TASK/12shapes each). Local sm120 stand-ins remove unsupported setmaxnreg and reduce CACHE/WS8 raw depth2→1; actual producer/math executes, and stand-ins are absent from H200. Exact packed local lab passed all16 seeds; new packet round-trip6633B, submitted source982951B. The H200 batch takes34s including18.8s compilation.

The evidence closes these exact dedicated-producer/cache/own-stream-L2 schedules as improvements over BASE. It does not close compute overlap generally. The next largest compute lever is reducing the416-instruction decode/scale body while retaining the eight-consumer register layout, rather than another pipeline-depth or L2 arm. Rank its cheapest SASS/resource proof first; a future structural batch should couple a genuine decode reduction with the proven producer as a separable contrast. The measured W2-ready/grid-barrier critical term remains open; no standalone sub1% run follows.

### P3a S19 integer/address issue and exactness premises (2026-10-26)

**`6f5afe0c273e4bbf`**, sender1006438, wrapper01a7a9c1aeed4525, completes16 fresh seeds at M1/8/16/32,128 paired ratios and64 A:A. Exact frozen S19 expanded62ea5bf4 is the BASE, including FAST+INDEX+SHARD2 M1, PAIR M8, PAIR+BLOCK M16/32 and pdl_M. Official host and all native TASK gates pass, no failures; max total/row .002456/.002491. Packet checksum, submitted source/cells, all12 runtime resource rows and seed completeness are verified. Candidate/control time; descriptive95% Student-t15 CI:

| Recipe | /S19 BASE [95% CI], M1; M8; M16; M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| AFFINE | 0.997260 [0.994542, 0.999978]; 1.000827 [0.996470, 1.005183]; 1.002168 [0.998112, 1.006224]; 0.997026 [0.994998, 0.999054] | Keeps row-valid/pass control; one 64-bit activation row base plus constant-offset vector loads, same16 .cg/.nc reads. M1 unchanged control alias, excluded; M32 point nominee. | Larger decoder cuts and other address/control schedules; no reduction of core decode instructions. | `6f5afe0c273e4bbf` |
| PRIVATE | 0.997059 [0.994298, 0.999820]; 1.008689 [1.003961, 1.013417]; 1.011149 [1.005550, 1.016748]; 1.015676 [1.011441, 1.019912] | Transfers steady TMA cursor from shared Route to issuing-owner registers after original initial copies; all8 math warps and2 raw slots retained. M1 point nominee; M8+ loses despite fewer static INT/shared sites. | P4 whole-layer R for M1; other producer-state representations without the added uniform/call-stack tradeoff, arithmetic/dynamic-lookup decode. | `6f5afe0c273e4bbf` |

A:A mean ratios at M1/8/16/32 are1.004719/1.003344/1.001305/.999876, SE .001485/.001874/.002525/.001268; mean absolute .174/.416/.508/.348us, max .672/.736/2.064/.976. M1 has positive A:A drift and its identical AFFINE alias also shows .997260 [ .994542,.999978 ]; PRIVATE M1 is therefore a weak point nominee, not established implementation benefit. The G0 per-shape point rule still hands it to P4 with this alias diagnostic; no A:A/CI veto or standalone confirmation run.

Per-pipe sizing: exact old457-site marker is273 INT (17LEA+112SHF+144LOP3),16 PRMT,128 HMUL2,32 HMMA,8 LDS. The scale LDS sits outside that marker. This corrects the earlier416 decode/32 MMA/9 LDS label, which mixed scopes; neither the kernel nor old timings change. All new arms retain that core exactly. Normalizing per16 requested weights/lane gives17.0625 INT,1 PRMT,8 BF16 multiply,2 tensor instructions plus.5625 weight/scale LDS. PIPE16c18078's empirical same-op envelopes1.96/1.96/3.57/.658/.25 imply these modeled SM cycles per unique weight before **and after** both arms:

| M | INT | PRMT | BF16 arithmetic | Tensor | Weight/scale LDS | Largest modeled issue demand |
|---|---|---|---|---|---|---|
|1 /8|.5441|.0319|.1401|.1900|.1406|INT|
|16|.6534|.0383|.1682|.2281|.1689|INT|
|32|.8654|.0507|.2228|.3021|.2237|INT|

Factors1/1.20098/1.59052 are historical logical weight-request repeats, not measured HBM misses. Actual saturation, exclusive scheduler stalls and a complete whole-kernel cycles/weight model remain unmeasured; uniform throughput is not calibrated. Do not add the separate operational envelopes as independent physical ports. Address/control changes are separately sized below.

Full-image INT/uniform/FMA/tensor site counts M1;M8;M16+ are BASE986/73/267/32;1799/66/292/32;1811/66/292/32. AFFINE986/73/267/32;1754/66/292/32;1769/66/292/32. PRIVATE989/71/267/32;1633/292/292/32;1647/292/292/32. M8+ PRIVATE replaces14 shared LDS/STS sites while adding226 uniform sites; these image counts include alternative/exception paths, not executed per-weight work. Marked finite B-fill BASE→AFFINE per-segment INT counts are31→11 (W13) and26→10 (W2); every form retains16 vector reads. With segment length L tiles, their INT contribution is .06178/L→.02192/L (W13) and.05182/L→.01993/L (W2) SM cycles per requested weight, before historical repeat factors. Core INT/FMA/tensor costs stay unchanged. PRIVATE's added uniform throughput is uncalibrated, so a predictive whole-kernel cycles/weight total is unavailable. PRIVATE all-M stack allocation8/16/16/16B is real, withzero spill stores/loads; BASE/AFFINE zero local. Every arm actually keeps2CTAs. Registers M1/M8/M16+ BASE117/126/126, AFFINE117/126/126, PRIVATE121/119/119; dynamic shared83904/84176/84496/85136B, unchanged.

INDEX's earlier d7fb2d8b B-fill31→13/26→8 cut changes uniform control across the full kernel: adds2 VOTEU,33 LEA,53 UIMAD,82 UIADD3 sites while removing others. Loads/spills/residency do not grow. This identifies added work before reusing the idea, not a unique runtime cause of the loss. AFFINE preserves the prior control; it does not reuse INDEX's branch hoist at M8+.

A:all4,064 finite signed E2M1 x E4M3 products/256 are exactly representable in FP16 and BF16. The usual single-affine magic-nibble form is not an exact E2M1 decoder (codes5/6/7 give2.5/3/3.5 instead of3/4/6). Exhaustive1,838,032 positive-BF16 magnitude-bit/base embeddings find only the incumbent zero-bias bits6/7/8 form. No proposed nonzero-bias HFMA2/HADD2 implementation is established; no A arm was sent or measured. Leaves open multi-piece arithmetic, other recodings and mixed conversion forms; this bounded result does not close FMA decode.

B:the bounded direct constant-index LOP3/PRMT zero-shift selection family cannot carry64 independent input-code bits through24 available source positions; four same-mask disjoint rotations also have no solution. No new nibble-permutation arm was established or measured. Leaves open dynamic-selector PRMT tables and arithmetic/layout combinations; these searches are not a general decoder lower bound. S19 already issues weight copies with bulk TMA, so simply adding TMA is not a new C arm. `structural/intpipe/premises.json` retains exact scopes and counterexamples.

P4 receives PRIVATE M1 (.997059, SE .001295,16 seeds) and AFFINE M32 (.997026, SE .000952,16 seeds) in `point_nominations.json` and `structural/intpipe/P4_HANDOFF.json`. Exact patches apply to the checksum-verified expanded S19: PRIVATE `private.patch` e0127e8da0a8e75ba9ceb0e6c362cb73f33a2200cfb5fc85b9698c7afa0f3ee3; AFFINE `affine.patch`4ebd938516334d5a79a69d968e86c76275f8f9b8594bb13d7699ded8676108cd. P4 confirms held-out six-load R; no robust or additive gain is claimed.

Static-first checks, all changed native sm90a units and import transport smoke pass;324 native TASK/current-input/capture checks pass without stand-ins. Exact wrapped local16-seed packet passes128 pairs64AA,5794B/81.1s. A send preflight first rejected a missing top-level lab entry before quota. The final queue command continued after a stale fingerprint report while its wrapper rerun finished; expanded body was unchanged and final full admission now passes. The actual H200 packet is complete,25.5s including compile. No failed test is suppressed.

The core per-weight decode lever remains uncut. Next by ceiling is a genuinely cheaper exact dynamic-selector/arithmetic decoder, proved on all4,064 products and counted against the unchanged S19 math body before another run. These small address nominees travel in P4's next composition; no standalone sub1% confirmation.

### P3a S19 W13→W2 dependency bound and producer-owned slices (2026-10-26)

**`554c392d10f54b71`**, sender1908836, wrapper458755c365021f20, completes16 fresh seeds at M8/16/32:48 matched NOWAIT oracle pairs,48 A:A and240 correct-build pairs. Frozen S19 expanded62ea5bf4 is BASE. Packet checksum,submitted wrapper/embedded payload/all6 source cells,all27 runtime resource rows and seed completeness are verified. Every correct arm passes TASK before/after timing;max total/row .002384/.002482,no failures. The lab-only host deliberately stops at prepare after emitting evidence;its `passed=false` is the transport's planned stop,not an experiment control failure. Native checks and complete packet establish the stage results. Candidate/control time;95% Student-t15 CI over16 seed-level paired medians:

| Recipe | /S19 BASE [95% CI], M8; M16; M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| NOWAIT oracle,outputs ignored | 0.932068 [0.924024, 0.940112]; 0.939360 [0.930836, 0.947884]; 0.936656 [0.926623, 0.946689] | Replaces only W2-ready and grid-generation spins with one acquire each;identical math/weight work,stores,fences,publish/election/reset. Timing only,never a candidate. Saves6.79/6.06/6.33% or5.19/4.54/6.83us,material at M16/32. | This is the matched wait-removal counterfactual,not a universal lower bound on algorithms that also alter traffic/reduction. Old critical-CTA waits11.6+5.0us overlap useful work elsewhere;they are not additive whole-MoE savings. | `554c392d10f54b71` |
| A64 | 1.453253 [1.404139, 1.502367]; 1.469402 [1.404037, 1.534766]; 1.370064 [1.340562, 1.399566] | 64-channel producer-owned intermediate slice; immediate W2,64 FP32 atomic partials,2 W2 raw slots. Removes expert-ready and in-kernel grid spins; loses. | Coarser slices and on-chip/DSM reduction; this closes only64-channel global-atomic scheduling,not dependency removal generally. | `554c392d10f54b71` |
| A128 | 1.320506 [1.269739, 1.371272]; 1.281139 [1.215064, 1.347213]; 1.252181 [1.204255, 1.300108] | 128-channel slice,32 atomic partials,2 W2 raw slots. Halves reduction count versus A64; loses. | DSM/cooperative reduction and fused norm/combine; unchanged weight bytes but4x W2 TMA transactions remain. | `554c392d10f54b71` |
| P128 | 1.461285 [1.401091, 1.521479]; 1.432148 [1.364353, 1.499942]; 1.342472 [1.296661, 1.388283] | Same128-channel/2-slot main schedule; separate FP32 slice buffers replace atomics,then timed reduction/norm. Loses. | Coarser/on-chip partials that avoid16.6/33.8/65.1MB logical partial write+read traffic; no general split-K refutation. | `554c392d10f54b71` |
| A128D | 1.297920 [1.252023, 1.343818]; 1.252912 [1.191335, 1.314489]; 1.232193 [1.183630, 1.280757] | 128-channel atomic slices with8 W2 raw slots in the original72KiB raw scratch. Best tested build but still loses. | Cluster DSM or owner-output accumulation with fewer global reductions and finish launches; two-CTA capacity alone does not guarantee occupancy. | `554c392d10f54b71` |
| P128D | 1.423436 [1.367753, 1.479119]; 1.395200 [1.329342, 1.461059]; 1.314272 [1.268209, 1.360334] | 128-channel separate partials with8 W2 slots; capacity restored,finish reduction remains. Loses. | A DSM/fused reduction without the22.3/23.4/26.2us finish cost; no evidence this global-partial schedule can beat S19. | `554c392d10f54b71` |

A:A mean ratios M8/16/32 are1.000455/.998558/1.000881,SE .001544/.001552/.001093;95% CI [.997164,1.003745]/[.995250,1.001867]/[.998551,1.003210]. Mean absolute differences .381/.414/.328us,max .848/1.120/.896us. These describe control spread and do not veto point nominees. None of the15 correct per-M point estimates wins;`point_nominations.json` and `structural/dependency/P4_HANDOFF.json` retain an empty new-winner list plus the exact evidence. All five checksum-pinned S19 patches remain in the manifest;no whole-layer R or additive gain is claimed.

Each CTA owns(expert,8-token block,intermediate slice),computes full4096-K W13,rounds SwiGLU to BF16 shared,and immediately multiplies that slice into all4096 W2 outputs. No whole-expert readiness or in-kernel grid spin remains;ordered finish kernels still require main completion. Timed zeroing,GS2 scaling,RMSNorm/combine and all input-dependent math are retained. Prepare only losslessly permutes raw W2 code/scale bytes;full-bank inverse proof passes. Weight bytes and aggregate HMMA work match BASE,while S128 W2 TMA transaction count grows4x(total W13+W2 calls2x);S64 grows8x(total3.33x). This larger copy/address/reduction structure is part of the implementation,not an isolated wait-removal gain.

H200 confirms BASE126regs,NOWAIT125regs and84176/84496/85136B shared at M8/16/32. A64 uses110regs/85648B;A128/P128108regs/86672B;A128D/P128D108regs/86768B. Every main has2-CTA capacity,zero stack/spills. Useful S128 work units average252/234/386 versus264 max grid CTAs,so capacity is not a claim that2 CTAs are always doing useful work. The D layouts recover72KiB W2 raw capacity with8 smaller copies without dropping residency.

Atomic/reduction cost is separately profiled on the first4 seeds(median3 cold replays each),not mixed into the unprofiled ratios. At S128,atomic main minus same-layout NORED main is7.97/7.32/8.52us(shallow) and9.70/8.29/10.39us(deep);zero+norm+combine adds9.34/9.86/10.48us or9.30/9.77/10.48us. Separate partials reduce main output-path contrast to4.27/4.24/6.50us(shallow),3.69/4.14/6.06us(deep),but their reduction/norm+combine costs22.60/23.46/26.06us or22.31/23.40/26.15us. These are net contrasts including addresses/scheduling,not exclusive atomic latency:NORED retains all40 S128 HMMA sites and live checksums but107vs108regs,same2CTA. Node sums omit graph launch gaps.

Observed local routes average15.8125/32.25/62.0625. S128 atomics execute2.073/4.227/8.135 million FP32 reductions;S64 doubles them. Separate S128 partials write+read16.58/33.82/65.08MB logically;atomic zeroing writes1/2/4MiB. Physical HBM/L2 traffic is unmeasured. Even deep NORED main is84.72/74.62/105.78us versus BASE79.02/75.96/108.06us and NOWAIT72.94/72.38/100.91us in the same profile subset;there is no measured path from deleting atomics alone to the oracle. This rules out treating either16.6us wait sum or atomics as a complete additive explanation.

Local admission:675 native TASK/current-input/capture checks with no stand-ins,including M9/M24,empty/zero/all-local routing,local starts0/96/224 and changed captured inputs. A128 normal,P128 forced,P128D forced full S19 screens each pass13TASK/12shapes with native new-leaf eager/capture calls;existing Hopper attention glue uses explicit local stand-ins. Static-first/changed-sm90a/transport tiers pass. Exact packed16-seed local branch coverage passes240pairs48oracle48AA,7081B packet. Before quota,the local packet exposed an oracle index bug and a deep-module resource-method lookup;both fixed and rerun. Host CUPTI Kernel9 header requires the judge image's Kernel10 recorder header. H200 completes25.4s including11.3s compile;none of those instrumentation issues reached the check.

Next by ceiling:dependency removal alone offers about6%,not the critical-CTA22% sum. This closes only the five64/128-channel global reduction layouts. Before another H200 run,size a cluster-local/owner-output reduction with coarser slices and fused norm/combine against the2-CTA and useful-unit budgets;it must avoid millions of global reductions,the extra TMA issue,and10-26us finish work. Coarser slices/DSM are unimplemented/unmeasured,not promised gains. Keep the separate M32 load-issue/data-movement ceiling open;cheap decoder searches remain exhausted and modeled INT saturation remains unproven. No standalone sub1% follow-up is sent.

### P3a remaining decomposition and sized parking case (2026-10-26)

No new performance arm or H200 run. Exact evidence/model/counts are in `dev/nt1_timeline/structural/dependency/{SIZING.md,sizing.json,sizing_counts.json}`. This is an odds/size parking decision,not a DSM impossibility proof or a new G0 gate. The requested1% is this task's work-sizing threshold. Times us;rows overlap and must not be added:

| Term | M8 | M16 | M32 | Measured / modeled scope |
|---|---:|---:|---:|---|
| S19 correct routed stage |75.422|72.229|105.042|Measured16seed event means,554c392d;router included identically |
| Pure stream |49.440|50.838|53.224|Measured historical D4,360810b1,not a matched S19 floor |
| Same-depth loads-only |47.586|53.740|69.284|Measured historical C;suppresses activation/MMA/readiness/finalizer |
| Historical correct minus loads-only |17.300|22.192|35.411|Measured contrast,not exclusive exposed compute |
| S19 dependency removal |5.192|4.539|6.829|Measured matched oracle;95% CI[4.329,6.055]/[3.658,5.420]/[5.367,8.291] |
| W13 first issue→math |2.208|2.208|4.000|Measured instrumented c49 criticalCTA seed0,20ef2355;includes setup/scheduling |
| W2 first tile |unmeasured|unmeasured|unmeasured|No retained W2 events;normal S19 first-phase split also unmeasured |
| Last-tile math,each phase |~1.024|~1.024|~.992|Model from steady W13 tiles2–8;actual W13/W2 last endpoints unmeasured |
| Warp0 accumulated epilogue |6.432|7.296|9.760|Measured taxed4seed CTA means;W13/W2 split unmeasured;per-tile .292/.292/.296us is a model |
| S19 residual after historical stream+matched wait |20.790|16.852|44.989|Cross-cohort arithmetic model;unassigned route/fill/drain/issue/activation/compute/epilogue |
| Exclusive compute not hidden under loads |unmeasured|unmeasured|unmeasured|No receipt separates all of those terms |

Clock observer ratios1.126/1.144/1.137 preclude treating individual subclocks as normal S19 phase costs. Earlier c768a8e3 M32 criticalCTA36.05W13+11.58ready+17.57W2+4.98grid+1.17combine is historical and nonadditive across CTAs. Its first issue/land and end-drain do not measure per-phase S19 fill/drain. Correction:paired-ratio mean times control mean is not mean candidate time;the measured360810b1 C means are47.586/53.740/69.284,not48.35/54.90/69.98. The old A-C differences cannot all be labeled exposed compute.

| Sized form | Ratio / cost versus control | Mechanism / decision | Leaves open | Receipt / evidence |
|---|---|---|---|---|
| Full local output partial then DSM gather | Timing unmeasured;>=204800B shared gives1CTA |128KiB partial+live72KiB weight slots fails2CTA;not built | Streaming/owner-sharded partials;this does not close DSM | count_sizing.py,CUDA13 calculator;554c392d resource basis |
| C16/S256 owner-sharded DSM | Timing unmeasured;108regs/no spills coarse probe,97168B modeled shared gives2CTA;output-path proxy4.85/4.14/5.20us | Complete4096K per expert/token-block cluster,16 ranks(opt-in),1.036/2.114/4.067M DSM reductions;modeled R1.002063 before sync/underfill/finish;not admitted | Faster DSM or additional measured compute/issue savings;actual cluster occupancy/additional registers unverified | sizing_counts.json,sizing.json;554c392d contrast scaled by count |
| C8/S512 owner-sharded DSM | Timing unmeasured;108regs/no spills coarse probe,109456B modeled shared gives2CTA;proxy2.42/2.07/2.60us |8 ranks hold completeK;.518/1.057/2.034M reductions,useful grids63/58.5/96.5CTAs;modeled R1.009286 before other costs;not admitted | Favorable half-proxy+.5us sensitivity R1.011396 remains open;half-proxy+1us R1.009840;neither is a measurement or impossibility proof | sizing_counts.json,sizing.json;554c392d count-scaled proxy |
| Fused finish only on A128D | Timing unmeasured;even all finish free leaves main15.40/6.95/8.10us above BASE in same profile subset | Cannot rescue unchanged slower main;S19 already has inline norm/combine,not three separate finish launches | New DSM dataflow plus completion publication/fusion;not a general fusion rejection |554c392d first4seed profiles |

A tile-streamed DSM gather fits in principle but32 output chunks need>=64 buffer-reuse cluster barriers plus norm synchronization;the2.127us total incremental budget would allow<=33ns/barrier if every other new cost were free. Barrier/network cost is unmeasured. CTA occupancy is not cluster occupancy;NVIDIA requires cudaOccupancyMaxActiveClusters and DSM lifetime synchronization. Coarse owner grids average126/117/193CTAs(C16) or63/58.5/96.5(C8) across132SMs,so2CTA resource capacity does not ensure useful residency. A rank can process multiple smaller slices,but retains their copy/reduction issue costs.

Planning load times v1/d16/d32 are126.97/176.09/221.25us from official S18 weighted times28b5c189 transported by paired S19/S18 load ratios6ac9317e. Full wall-time credit for5.19/4.54/6.83us gives wait-only R1.016674;>1% permits about2.127us common added cost across these three of six loads. This is not a universal DSM/NT1 ceiling. The count-scaled global output-path proxy is not measured DSM throughput or a lower bound;its original4seed contrast is noisy,especially M8. The favorable C8 sensitivity above1% is retained explicitly,but lacks calibration and margin for synchronization,coarse scheduling and finish. No supported>1% dependency/DSM build is identified under this historical estimate;the matched residual below is separate.

Do not admit these dependency/DSM builds on those cost estimates;the orchestrator rejected extending this decision to the small-M lever. Reopen on actual2CTA cluster occupancy plus a matched coarse/DSM/finish cost fitting the~2.13us common budget,or an additional measured compute/load-issue saving that increases it;an exact lower-issue decoder/layout or public/per-shape leader evidence can also reopen. No new quota/P4 nominee. Last best measured A128D/S19 remains1.297920[1.252023,1.343818]/1.252912[1.191335,1.314489]/1.232193[1.183630,1.280757],16seeds,554c392d;only those recipes close. P4's prior nominations and GOAL G0 are unchanged.

P12's03:18UTC snapshot97f5e7f3 compares S15 aggregate MoE6627.36us to leader5266.72us:1.25835x,about20.53% less component time demonstrated elsewhere. Leader per-shape times/implementation are unavailable. This could be heavy-M or other routing/tiling work;it neither measures S19's small-M gap nor identifies the recoverable mechanism. It keeps broader MoE opportunity open;the matched ladder below supersedes lane-wide parking.

### Matched S19 residual assignment: active, 5ee35b941ccb4171

The lane-wide parking conclusion is superseded. This single cohort uses unchanged expanded S19
62ea5bf4,264CTAs×256threads,depth2,identical route seeds/tiles/shared budgets. Eight seeds×four
symmetric ec754e5 rounds retain all2,048 times,96NVML snapshots and all work/finish histograms.
All60 runtime resource rows match sm90a counts,two-CTA capacity andzero spills. BASE/AA/BSINK/HIST/EARLY
TASK controls pass. The host's declared lab-only prepare stop is not a native correctness failure.
[Full means/95%CIs,contrasts and audit](../problems/muse_decode_layer/dev/nt1_timeline/structural/ladder/REPORT.md).
Ratios below are paired geometric time ratios versus exact S19 BASE,ordered M1/M8/M16/M32.
Diagnostic cuts have ignored outputs and cannot nominate. Wide absolute time CIs include route variation.

| Recipe | Ratio vs BASE (M1/8/16/32) | Mechanism / result | Leaves open | Receipt |
|---|---|---|---|---|
| L0 | 0.65999 / 0.76139 / 0.74272 / 0.67879 | Raw copies plus every raw shared read,checksum;no math/activation/finish | Load-side issue/repeats versus unique STREAM;not exclusive memory latency | 5ee35b941ccb4171 |
| L0S | 0.71661 / 0.79986 / 0.75161 / 0.73165 | Common volatile live sinks on raw path;calibrates sink tax | Sink scheduling differs from ordinary BASE | 5ee35b941ccb4171 |
| L1 | 0.72773 / 0.80823 / 0.78954 / 0.79309 | All exact decode/scale products live;no MMA | Cheaper decode forms;not proof of integer saturation | 5ee35b941ccb4171 |
| L2 | 0.83675 / 0.93812 / 0.96197 / 0.99604 | Adds activation operands plus all32HMMA sites | Separate operand transport from mandatory math/scheduling | 5ee35b941ccb4171 |
| L3 | 0.95876 / 1.00362 / 1.03559 / 1.06097 | Adds original epilogue,publication and NOWAIT finalizer | Correct readiness redesign must pay its reduction cost | 5ee35b941ccb4171 |
| BASE | 1.00000 / 1.00000 / 1.00000 / 1.00000 | Unchanged original S19 control | All matched residual mechanisms | 5ee35b941ccb4171 |
| BSINK | 1.06129 / 1.08100 / 1.10422 / 1.13864 | Original BASE plus common sinks;tax2.14/5.52/7.42/13.42us | Sink-free schedule costs are conditional | 5ee35b941ccb4171 |
| MATH | 0.93294 / 0.89564 / 0.89346 / 0.87692 | On-chip raw operands/register activations;small globals,initialization and output remain | Not a universal compute floor;operand layout/scheduling | 5ee35b941ccb4171 |
| STREAM | 0.64529 / 0.76547 / 0.66631 / 0.51760 | Two-slot TMA of exact touched experts unique bytes;simple tile loop | S19 raw issue/repeats versus ideal transport;no correctness claim | 5ee35b941ccb4171 |
| META | 0.16807 / 0.10933 / 0.10564 / 0.07891 | Route/metadata lists live with same grid/reservation | Timed route fusion and work mapping | 5ee35b941ccb4171 |
| ACT | 0.28525 / 0.21864 / 0.23515 / 0.21234 | Real activation transport,operand sinks once per segment | ACT−LOOP includes sink overhead;pack-free oracle needed | 5ee35b941ccb4171 |
| LOOP | 0.19333 / 0.14362 / 0.13917 / 0.10853 | Metadata plus tile schedule/partial-store barriers,no math | Activation reference must subtract this schedule cost | 5ee35b941ccb4171 |
| EARLY | 1.00304 / 1.01450 / 1.01998 / 1.02053 | Release slot/issue next copy after volatile g7 raw reads,warp sync,before last decode/MMA;M1 alias | Closes this placement only;operand layout,MMA scheduling and M32 load-side gap remain | 5ee35b941ccb4171 |
| AA | 1.00157 / 0.99940 / 1.00145 / 1.00119 | Independent workspace,same exact BASE;noise control | A:A is uncertainty,not a per-shape veto | 5ee35b941ccb4171 |
| HIST | 1.02018 / 1.00600 / 1.00874 / 1.01164 | Minimal observer of CTA tiles/experts/SM/entry/W2/end | Measured observer tax;tails overlap other terms | 5ee35b941ccb4171 |
| ROUTER | 0.18999 / 0.09897 / 0.09529 / 0.07015 | Separate actual router/input producer,not included in NT1 BASE | Timed packing/fusion must include producer cost | 5ee35b941ccb4171 |

Calibrated adjacent contrasts (us,95%CI),M8/M16/M32: decode L1−L0S
.539[.169,.909]/2.543[1.486,3.600]/6.022[4.821,7.223];largest MMA+activation L2−L1
8.785[7.464,10.106]/12.096[11.021,13.171]/19.705[16.719,22.691];epilogue NOWAIT
4.392[3.925,4.859]/5.207[4.766,5.648]/6.292[5.339,7.245];sink-bearing dependency
5.124[4.770,5.478]/4.838[3.743,5.933]/7.489[5.598,9.380]. M1's largest added term is
4.087[3.738,4.436]us epilogue. L0−STREAM M32 is15.695[11.719,19.671]us;M8 crosses zero.
Every seed telescopes exactly with both raw and normal sink calibration,not by mixing historical nodes.
These are conditional schedule contrasts,not independent hardware-pipe latencies.

Largest-term full-wall-credit six-load ceilings M1/8/16/32 areR1.00826/1.01202/1.01193/1.01567;
M8–32 MMA+activation togetherR1.04014,M32 raw-load side aloneR1.01234. BASE−STREAM
11.991/15.654/22.958/47.102us modelsR1.11685;one thirdR1.03532. Layer-time inputs are transported
S18→S19 planning data,not measured held-out G0. Work histograms retain8,448CTAs;W2 p50→last
1.184/4.062/5.732/7.964us,final-exit p50→last1.176/1.128/1.140/1.342us. These observer tails overlap
ladder terms. Seed0 additionally retains per-SM counts/last finish and joint finish bins at2.048us.

EARLY/S19 M8/16/32 is1.014503[1.008061,1.020986]/1.019984[1.011677,1.028359]/
1.020532[1.008815,1.032386],paired logSE.002633/.002800/.004884,A:A logSE.002694/.003458/.002561,
eight seeds0–7 with four arithmetic rounds each. M1 is unchanged alias1.003039,excluded. EARLY uses
128regs,zero stack/spills and84176/84496/85136 shared bytes,two CTAs. 135native TASK checks and
normal+forced13TASK/12shape integration screens pass;changed leaf remains native. Exact patch
ba90ff8e,candidate789a0223 in ladder/early/manifest.json. No new point winner or whole-layer confirmation;
P4_HANDOFF.json and point_nominations.json retain the result and all83previous nominees unchanged.

Next ranked bound: pack-free/coalesced BF16 activation operand layout,then timed producer packing
only if exposed benefit pays. ACT−LOOP reference5.021/6.810/10.043us includes operand sinks;modeled
combined R1.02126 leaves<3.57us common added cost for>1%R at full credit. A bank reserves at most8MiB,
16MiB for both phases,no assumed byte saving;keep two CTAs and BF16,all activation work timed.
This mechanism remains unimplemented/unmeasured. M32 load-side scheduling is another>1% opening.
The dependency/DSM forms previously sized stay unadmitted;the lever stays active.

Transport history:c43111f584854b05 is unmeasured(schema/output oversize);da1fb2eeb0d3443d has valid
controls but scalar STREAM/per-tile ACT sink measure costlier implementations,not pure-byte/activation
oracles. initial_* keeps that evidence. All rows above use only corrected5ee35b941ccb4171.

### P3a operand/issue oracle batch: queued, not measured (2026-10-26)

Read the public MoE P4/P3a handoff and matched-ABI mechanisms first: no whole-port/hybrid winner;
operatorstack's register-only raw loads and bit-permuted decode do not establish a transferable small-M
speedup. Read mixed_wgmma DIRECT/TRANS/WS/CTA2 losses and bounded CHUNK winners;this batch copies
none of their decoded-weight sharing/WGMMA producer-consumer designs. Compute/operand work ranks first.
Old MATH changes126→47regs atM8–32 plus operands/weight transport;max(MATH,STREAM) is not a proved
critical-path roof. No global hardware-pipe saturation is inferred from that comparison.

Source-derived issued warp-HMMA from5ee35b94's actual CTA tile/expert histograms and32SASS sites:
mean M1/8/16/32 **516096/1400832/1474560/2187264**;useful column fractions
**.125/.27145/.48769/.69069**. Per-seed counts and95%CIs are in operand/sizing.json. This is an exact
schedule-count reconstruction,not NVPerf counters. The useful equivalent does not implement N<8 BF16 MMA.
Wasted-HMMA nominal service at TASK989TFLOP/s with full wall credit models six-loadR1.01097;not measured
exposed savings. Prior ACT−LOOP referenceR1.02126 and epilogue referenceR1.01580 overlap this scenario.
Pure occupancy upside from lowering registers at fixed two CTAs isR1.00000;latency/scheduling stays open.

| Recipe | Ratio / us / CI | Mechanism | Leaves open | Receipt / sender |
|---|---|---|---|---|
| FRAG_SMEM / BASE_PAD | H200 pending M1/8/16/32 | Synthetic BF16 fragments prelaid in8KiB two-plane shared bank;segment-cached Bf,extra8320B reservation controlled | Real64KiB activation bank,packing/publication and capacity cost;not a production candidate | sender3637718,wrappere1710fb3 |
| FRAG_REG / BASE | H200 pending | Pack-free register constants remove gathers/addressing and lower lifetime;64regs atM8+ | Joint operand/register ceiling,not exclusive global-load latency | same pending batch |
| FRAG_JIT / FRAG_SMEM | H200 pending | Same shared pattern,one K-group live;63vs117regs,more LDS per tile | Correct cached/shared layout must pay its working-set and transport costs | same pending batch |
| ACC_FREE / ACC_CTRL | H200 pending | Both128regs atM8+,32HMMA and matched live-result sinks;FREE zeroes C to remove recurrence | Wrong math oracle only;independent correct accumulator scheduling | same pending batch |
| PAD_USEFUL / PAD_CTRL | H200 pending | Both retain all weight/decode work with matched sinks;only r of8 K-warps issue HMMA to match useful-N budget | Skips required K work;not a legal narrow-N implementation or nominee | same pending batch |
| L1/L2/L3 + HIST | H200 pending | Same matched decode/MMA/epilogue cuts and minimal work/finish observer | Epilogue resource127→128regs retains2CTAs;no occupancy-only saving | same pending batch |
| LAZY / BASE | H200 pending;native leaf only | Original BF16 activation loads moved per K-group/tile,126→96regs atM8+;repeats global reads | Correct but no production graph measured;stage point cannot establish S21 transfer | same pending batch |
| BF16 conversion removal | Dropped before timing:zero convertible instructions in this fill | S19 loads BF16 bits directly into uint4/Bf for HMMA | Fragment/address/lifetime work still exists | actual frozen S19 source62ea5bf4 |

All56 exact sm90a arm/shape counts give2CTA/zero stack/spills;all HMMA-bearing arms retain32sites,
L1 none. ACC_FREE's first compiler output collapsed to one HMMA;matched live-result sinks fixed this
before send. Real full Bf cache65536B plus two raw slots needs150672B atM32 and fails2CTA;one slot
fits113808B,but repeated shared loads/publication and the known one-slot penalty must be paid. Await
the FRAG_JIT bound before admitting that build. Original read/publish/acquire ordering stays in all arms.

Exact payloadf4a1f883:448native control TASK checks plus162LAZY native nearby/local_start/skew/changed
captured-input checks pass (leaf worst total/row.002392/.002451). Local packet8985B fits11700B;all1920
rounds,96NVML snapshots and work/finish records retained. Local sm120 timing is not H200 evidence.
All five hosts exhausted quota;actual MoE check sender**3637718** waits for**10-27 09:00KST**,
wrappere1710fb3ff91fcb6a430acf5bede584ae4172ae08188823f94d673034f636301,no judge job assigned yet.
A prior outer-payload AST admission error spent no judge quota;literal lab wrapper was fixed and both
exact native screens rerun. See [REPORT](../problems/muse_decode_layer/dev/nt1_timeline/structural/operand/REPORT.md).

Decision remains active. New oracle timings,CIs and the largest buildable term are pending the receipt.
Production base pinned to S21 expanded109c0135/shipmenteacc087b. After ranking,include all timed
packing/producer cost and pair the correct build in its complete12-shape production graphs with fresh
seeds under promotion.evaluate;TASK only and held-out six-loadR lower bound>1 judge it. No per-call
transfer or new G0 claim;all83previous nominations remain unchanged. No production build is declared measured.

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
| DIRECT serial, S3 for MAX64 |1.1976 /1.2273|Improves serial by~4.9/5.5%, still loses to S14; does not establish S3 on parallel CTA2|Repaired CTA2S3 direct pair; another pipeline/ownership|`8792099636654bac`|

`8792099636654bac` controls: S14 182.90/256.67us, A:A signed/absolute/max difference
+0.043/0.519/3.120us across both M, worst FP32 L2/row0.004134/0.004709. All arms48/48
bit-equal at each M. Ordinary DIRECT's ratio1.1965/1.1381 agrees with the prior1.1970/1.1322
calibration (`8ee003db`) within0.6%; isolated heavy-phase totals also remain within~2%. The measured
loss closes these topology changes, not decode-once itself. Do not allocate the old residual
into causal decode/epilogue/overlap components from isolated phases and a different mean input.

P3b round2-5 receipt decisions (same candidate/S14 convention, M128 / M256):

| Recipe | Paired ratio | Mechanism / evidence | Leaves open | Receipt |
|---|---|---|---|---|
| TRANS64, tokens register-A / decoded weights shared-B |1.4350 /1.5270|Measured loss; fixed heavy+pack182.69/303.23us vs152.73/249.68 predicted; pack alone9.82/10.54us|Cheaper publication/layout without timed fragment packing|`dd539517bc8344dd`|
| TRANS128, two token WGs sharing decoded-B |1.7416 /1.7781|Measured loss; one CTA/SM; fixed heavy+pack203.62/377.36us vs93.48/150.23 modeled|Another bounded-width ownership without the issue/occupancy cost|`dd539517bc8344dd`|
| CTA2 / CTA2S3, adjacent-output WGs sharing X |**Unmeasured** (raw1.3423/1.2973 and1.2066/1.1601)|Numerical gates pass, but missing all-consumer X retirement join; per-WG wait does not retire the other WG; all full-width classes one CTA/SM|R6 repairs join and tests real two-CTA bounded64-token chunks; raw time cannot close this recipe|`dd539517bc8344dd`|
| Compact expert-major STATIC / QUEUE / QUEUE_S3 tail |**Unmeasured**, no valid pairs or A:A|First native M128 fixture L2.359844/row1.0; generated dispatcher omitted BF16 N16, portable stand-in supported it. Shared-X join also unproven|R6 supplies native N16 and retirement join; scheduling itself remains open|`b93dbbadd1014984`|
| WS_TILE_BARRIER, TMA raw ring /3 decode warps /2 RS consumers |2.0471 /2.0381|Measured loss; one CTA/SM confirmed; generic decoded-fragment staging and gated epilogue|Other specialization/publication/ownership forms; no isolated phase proves one culprit|`b93dbbadd1014984`|
| WS_PIPE, same ring with cross-tile stream-before-epilogue |2.0320 /2.0248|Measured loss; raw mean saves2.83/3.32us versus barrier but does not repay the~190/260us loss|A cheaper pipeline without this staging and register footprint; specialization generally remains open|`b93dbbadd1014984`|
| Fixed two-N32 token groups, RS raw-share |1.3017 /1.3472|Measured loss despite confirmed2CTAs/SM,122/123regs; padded decode/MMA work grows1.6/1.455 vs S14|Bounded64-token adjacent-output WGs preserving S14 token padding|`7ea5e070b8f744ea`|
| Same token groups, SS decode-once |1.4114 /1.4931|Measured loss despite2CTAs/SM; decoded-A writes402.65/805.31MB plus two shared reads, source cost not isolated|Cheaper decode sharing without this publication/padding form|`7ea5e070b8f744ea`|

Round2 controls183.56/254.94us,22 seeds x3/M, A:A signed/absolute/max−.051/.428/2.448us.
Round4 controls184.67/256.38us,24 seeds x3/M, A:A−.039/.421/1.936us. Round5 controls184.30/
255.61us,24 seeds x3/M, A:A−.105/.498/3.408us. All qualified new-arm TASK gates pass, worst
L2/row.004134/.004756. Round4/5 primary S14 pairs remain valid; their secondary CTA2 references
are unqualified. Round3 fails before summary and reports no A:A; its failure does not invalidate
the separately completed round4 payload in that packed run. No arm is nominated to P4.

Model audit: preserve original predictions, normalize forecast to the current S14 control before
scoring. TRANS64 whole-MoE260.2/381.1us vs262.7/388.2 gives error factor1.020/1.043, but fixed
heavy service is~20% underpredicted, so whole-point agreement contains cancellation. TRANS128
201.0/281.7us vs318.8/451.3 gives1.602/1.641. Warp PIPE187.4/193.9us vs374.6/517.4 gives2.008/
2.716: fully hiding its stage residual under a4TB/s stream is falsified for this implementation.
Withdraw the unmeasured25/50% queue recovery and do not fit unsafe CTA2 raw timing. Resource
occupancy is verified input, not a throughput forecast; isolated services and different route
means cannot identify causal decode, epilogue, launch, tail or interbranch overlap components.

Applied follow-up **R6 bd2192e2 / CUDA8c476193**, sender **50c82e7edb5e83aa**, job **b433947196f94ef4**, MoE import150s:
safe full-width CTA2S3/static/queue/queueS3 plus bounded64-token S3/S4 two-CTA targets. Six forced
official screens,1729 captured checks and1380 source-bound protocol checks pass; native SASS
now requires BF16 N16/32/64. The old checker incorrectly assumed the missing retirement join;
negative witnesses now reject it. R5 checker now captures immediately per plan to preserve its
RS/SS mode;1036 checks pass again, and the actual sent b610039e payload already did this.
R6 native pairs are recorded below. Its first oversized wrapper was rejected before a job; retry
uses the H200-gated round2 host prefix, unchanged payload,275130-byte wrapper. Evidence and
current model: `problems/muse_moe_w4a16/dev/mixed_wgmma/branches/review/` and `round6/`.

P3b R6 retirement/N16 repair and bounded-token receipt (ratio/S14, M128 / M256):

| Recipe | Ratio | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Safe full-width CTA2S3 |1.2196 /1.1629|Both consumers join X retirement; valid TASK, oneCTA/SM retained|Bounded token width with lower accumulator pressure|`b433947196f94ef4`|
| Safe compact STATIC |1.5266 /1.4609|Native BF16 N16 present and retired shared-X ring; valid loss|Other expert-major ownership; original missing math stays unmeasured|`b433947196f94ef4`|
| Safe compact QUEUE |1.5463 /1.4749|One ticket per expert/channel tile, correct N16 and ring; valid loss|Queue granularity and lighter dependency protocols; static loses too, so atomics alone are not isolated|`b433947196f94ef4`|
| Safe compact QUEUE_S3 |1.2587 /1.2428|Deeper ring helps this image, still a valid loss|A cheaper bounded-width schedule|`b433947196f94ef4`|
| CHUNK64 S3, adjacent-output WGs share X |1.0165 /**.9448**|128regs,76800B ring, native2CTAs/SM; nominateM256 point gain|Whole-layer overlap and held-out six-load score; no M128 point gain|`b433947196f94ef4`|
| CHUNK64 S4, same bounded ownership |**.9791 /.9083**|181.55/233.26us vs185.92/257.34;102400B ring, native2CTAs/SM; nominatebothM point gains|P4 composition/confirmation; this net result does not isolate bytes versus occupancy or wait effects|`b433947196f94ef4`|

R6 all24seeds x3/M pass TASK, worst L2/row .004134/.004756, A:A signed/absolute/max−.159/
.429/1.712us. Original summary omitted SE; means/extrema cannot recover it. All three eligible
arm/shape point pieces are handed toP4 with that limitation and a labelled SE bound; no
per-load significance or A:A veto. Earlier qualified round2-5 primary S14 means are all>1,
so no older verdict changes. Exact root-applicable S15 patches and normal/forced whole-layer
screens: `branches/s15_review/P4_HANDOFF.json`; fresh S15 follow-up retains seed means/SE.

Attribution correction:182/256us is whole MoE, not heavy. Stage3 fixed isolated S14 heavy
W13/W2=43.52/34.94 and80.46/47.86us, light81.17/49.74 and71.14/44.14. Neither isolated sums
nor different-input means are an additive critical path. S15's NT1 completion chain overlaps
other MoE work; actual individual wait/decode/epilogue clocks are not retained. Native O3
S15 KT128 loop counts188 total instructions,116.25 integer/address and32HMUL2 per thread;
static counts do not identify stall time. Full evidence and each rebuild's measured cost:
`branches/s15_review/ATTRIBUTION.md`. Withdraw free-hidden-residual and arbitrary queue credits.

New S15 premises confirmed: heavy W2 follows the entire W13 grid on one stream but reads its
own expert/32-token ACT block; there is no heavy `done[e]` counter. READY adds128 tile
publications/block with all-writer fence/join and consumer acquire; matched sequential grid
control isolates capacity. PACK5 preserves the exact wide/interleaved code bank, reverses
PERMW only for byte coding and reconstructs it in timed decode. Its full805306368 code bytes
and100663296 scale bytes prove exact,4.05816% net saving. Native PACK5 stage image188→309.75
instructions includes alternative raw/compressed paths; zero spills,103/104regs. P3a's fresh
PACK5 losses are an adverse prior, not proof of this different heavy path. The two-base37→406
recipe is dropped. Their S15 paired receipt is recorded below.

P3b isolated S15 confirmation `7e38226389a848eb` (24 fresh independent weight seeds,3fixtures/seed/M):

| Recipe | M128 ratio ±SE / M256 ratio ±SE | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| CHUNK64 S3 on exact S15 NF |1.032764±.007501 / **.946036±.013047**|Native2CTAs/SM; valid N32/N64; nominateM256 point piece|P4's held-out six-load composition, layer overlap differs from component|`7e38226389a848eb`|
| CHUNK64 S4 on exact S15 NF |**.997500±.009854 / .909681±.007858**|175.862/219.844us vs176.552/241.940; native2CTAs/SM; nominatebothM despiteM128 uncertainty|P4 score gate; freshM128 gain is.25%, smaller than R6 discovery2.1%|`7e38226389a848eb`|
| Exact heavy PACK5 |1.168932±.006967 /1.260287±.006467|4.05816% fewer exact bytes but added timed scale unpack/metadata; native4CTAs/SM retained,104/103regs,no spills; loss is not an occupancy drop|Cheaper exact coding; net test does not isolate decode from metadata cost|`7e38226389a848eb`|
| Matched sequential W2-grid control |1.039774±.009193 /1.155520±.013055|Same S15 kernels, W2 grid limited toSM count; exposes capacity cost|An uncapped readiness design preserving memory-level parallelism|`7e38226389a848eb`|
| Own token-block READY versus S15 |1.074708±.006290 /1.074582±.005278|Correct per-block publications/acquires; Kloop unchanged, native4CTA resource admission but capped consumer grid; net loss|Readiness without this grid cost; same-seed normalized READY/serial contrast~1.0349/.9324 is exploratory, not separately timed ABBA|`7e38226389a848eb`|

AllTASK gates pass, worst L2/row .004128/.004955. A:A signed/absolute/max on24 seed means
(each averages3fixture pairs) is+.100/.253/1.653us atM128,−.098/.306/.874 atM256; max here is
not a max over individual fixtures. Ratio SE is across independent weight seeds. Binary
transport retains all seed means, no dropped lines. S4/M128 remains nominated even inside
A:A/SE uncertainty; no older round2-5 qualified point winner was excluded. P4 gets all3
eligible arm/shape pieces with actualSE and exact root-applicable patches in
`branches/s15_review/P4_HANDOFF.json`, then confirms held-out six-load R. Neither packet
nor capped-readiness losses close their general families. Normal/forced whole-layer screens
all10pass (13TASK,12shapes,26static),684 captured checks are bit-equal toS15 across57 cases,
9raw-scale graphs pass, and7458 timed-decoder packed/raw groups preserve all u8 scale codes.

### P3b matched stream / count-routing verdict (2026-10-26)

`b5c4d99ed0ed4dc6`:16 fresh independent weight/router seeds,3fixtures/seed/M,
all7 arms pass TASK before/after timing (maxL2/row .004122/.004719). Direct cold
pairs against both S15 and CHUNK4; A:A signed/absolute/max seed-mean us−.014/.329/.683
atM128,+.016/.315/.843 atM256. Native CH4/balance both128regs,102400B shared,2CTAs/SM.
Ratios below are new/CH4 time, descriptive95% Student-t15 CI; all S15 contrasts,
SE and seed samples are retained in `branches/balance/results.json`.

| Recipe | M128 /CH4 [95% CI] | M256 /CH4 [95% CI] | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|---|
|B8|1.000386 [0.996014,1.004758]|1.001866 [1.000737,1.002995]|Masked NT1 counts, same ownership; CH4 math unchanged. No current-cell point win; /S15 .974856/.916912 alternatives to P4.|More general expert-dependent ownership; activation reuse and service reduction with preserved residency.|`b5c4d99ed0ed4dc6`|
|B4|0.998493 [0.986834,1.010153]|1.013878 [1.008315,1.019440]|Move counts5–8 NT1→N16. M128 .15% point gain to P4 despite CI crossing1; /S15 .972042/.927495. No standalone small-arm run.|More general expert-dependent ownership; activation reuse and service reduction with preserved residency.|`b5c4d99ed0ed4dc6`|
|B2|1.030728 [1.007399,1.054056]|1.035302 [1.022187,1.048418]|Move counts3–8 toN16. Loses /CH4; /S15 .950139 atM256 is an explicit slower alternative to P4.|More general expert-dependent ownership; activation reuse and service reduction with preserved residency.|`b5c4d99ed0ed4dc6`|
|B0|1.216884 [1.185588,1.248180]|1.147339 [1.130234,1.164445]|All counts1–16 useN16; NT1 launches removed. Loses21.7/14.7%; eliminates NT1 tail without recovering the gap.|More general expert-dependent ownership; activation reuse and service reduction with preserved residency.|`b5c4d99ed0ed4dc6`|
|ONE|1.247776 [1.226239,1.269313]|1.249694 [1.237685,1.261703]|One N32/N64 CHUNK4 path for all experts, no light math launches. Padding/consolidation loses24.8/25.0%.|More general expert-dependent ownership; activation reuse and service reduction with preserved residency.|`b5c4d99ed0ed4dc6`|

Matched ordered oracle, conditional us M128/M256 (same first fixture per seed):

| Measurement | S15 | CHUNK4 | Mechanism / leaves open | Receipt |
|---|---|---|---|---|
| Actual unique stream U |114.411/131.750|same|~4.16/4.22TB/s; per-seed4TB model118.997/138.904. Earlier182.9/256.7 means and120.3/127.4 floors were from different fixtures.|`b5c4d99ed0ed4dc6`|
| W−U |3.574/12.659|.458/3.540|Extra scheduled weight requests, with stream cache reuse; not raw DRAM rereads.|`b5c4d99ed0ed4dc6`|
| X−W |39.302/99.882|16.269/43.810|Valid activation requests plus changed reader work/issue and L2 reuse. Activation reuse across more output tiles is open.|`b5c4d99ed0ed4dc6`|
| B−X |−1.408/−14.179|3.053/−3.427|Three-branch/two-phase launch of identical requests has no consistent large penalty; not a causal real-branch imbalance tax.|`b5c4d99ed0ed4dc6`|
| R−B |21.641/14.474|39.026/48.490|Restores actual math, waits, real memory schedule, epilogue, route/finish; individual service components remain unseparated.|`b5c4d99ed0ed4dc6`|

Full matched R15/R4=177.520/173.217us atM128,244.586/224.163 atM256. The ledger
sums to58.523/105.682us above the per-seed4TB models for S15,54.220/85.259 forCH4;
U−model=−4.586/−7.154us. Differences include interactions, not additive causal
stall buckets. Logical X requests281.63/595.03MB(S15),168.87/325.68(CH4) are not
HBM counters. External timestamps perturb totals5–8% (measured marked/unmarked
control); NT1 finishes last in marked graphs, but exact unmarked branch/wait gaps
remain unmeasured. Unmarked production ratios and oracle pairs are valid.

CH4 repeats /S15 .975053[.958368,.991739]/.916849[.904303,.929395]. Targets132.6/
158.5us remain open. `branches/balance/P4_HANDOFF.json` nominates every valid S15
orCH4 point gain with both controls and seed CIs; root S15 patches B4`b84a2255`,
B8`de11ef09`,B2`1f82d368`. Default B4/M128,CH4/M256; preserve S18's P7 W2/finish
shell and confirm held-out six-load R. Next large recipe: share each X stage across
four output tiles with two consumer WGs, checking register/shared admission first;
conditional half-X ceiling~8/22us does not hide the remaining39/48us service/shell.

### P3b S18 activation-reuse verdict (2026-10-26)

`4e1926c6185e4628`,16 fresh independent weight/router seeds,3fixtures/seed/M:
all seven math arms pass native TASK before/after, worstL2/row .004133/.004738.
A:A signed/absolute/max seed-mean us +.088/.323/.763(M128),+.167/.353/.960(M256).
Runtime CUDA occupancy confirms2CTAs/SM for all six variants. Native TMA/multicast
protocols are measured and correct. Ratios are whole-MoE new/S18, descriptive95%
Student-t15 across seed means; no A:A or per-load significance veto is used.

S18 controls173.140/226.723us and logical activation requests186.205/362.544MB.
Local sizing used16seeds x3fixtures and retains every expert's count/bytes; native
fixtures differ, so the table uses per-native-fixture counts. These are global
requests, not HBM counters; gather read/write and padded TMA boxes are included.

| Recipe | X MB M128/M256 | /S18 M128 [95% CI] | /S18 M256 [95% CI] | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|---|---|
|PACK|188.890/368.780|1.096557 [1.088904,1.104209]|1.073146 [1.068733,1.077559]|Timed parallel heavy-X gather plus contiguous address generation; no point gain. Original8x16B lanes already fill aligned128B spans.|Activation reuse without all-rank stage barriers,smaller useful grids or register serialization;partial fusion remains open.|`4e1926c6185e4628`|
|TMA1|231.948/444.835|1.284546 [1.254956,1.314137]|1.350137 [1.312048,1.388226]|Same gather plus single-CTA-cluster TMA ring/barriers. Loses;this controls the whole transport/sync recipe,not intrinsic TMA latency.|Activation reuse without all-rank stage barriers,smaller useful grids or register serialization;partial fusion remains open.|`4e1926c6185e4628`|
|MC2|145.964/257.140|1.299130 [1.265084,1.333177]|1.365487 [1.321498,1.409476]|Same bounded token tiles,2CTA output-panel multicast,per-stage all-rank arm/retire barrier. Lower requests lose;all native gates pass.|Activation reuse without all-rank stage barriers,smaller useful grids or register serialization;partial fusion remains open.|`4e1926c6185e4628`|
|MC4|102.973/163.292|1.328523 [1.288312,1.368735]|1.385554 [1.342586,1.428522]|4CTA output-panel multicast with same math/padding;largest byte reduction but loses32.9/38.6%.|Activation reuse without all-rank stage barriers,smaller useful grids or register serialization;partial fusion remains open.|`4e1926c6185e4628`|
|W4N32|121.750/212.876|1.300076 [1.221830,1.378323]|1.695647 [1.565819,1.825475]|Two64-output tiles per WG share X;2CTAs,zero spills. Smaller output grid and heavy N32 blocks raise weight requests478.94→540.87 /609.29→770.31MB.|Activation reuse without all-rank stage barriers,smaller useful grids or register serialization;partial fusion remains open.|`4e1926c6185e4628`|
|W4N64|121.750/212.876|1.222323 [1.122675,1.321971]|1.380142 [1.277689,1.482596]|Same shared-X reuse,original token blocks/weight counts.2CTAs,zero spills,but C7512 WGMMA serialization at127–128regs and smaller output grid.|Activation reuse without all-rank stage barriers,smaller useful grids or register serialization;partial fusion remains open.|`4e1926c6185e4628`|

All6 native layouts fit2CTAs:PACK102400B dynamic124/128regs;TMA1/MC2/MC4 add1024B
static,124–128regs;W4N32 106496B111/112regs,W4N64 104448B127/128regs. Admission is
not proof of two useful heavy CTAs on every SM; output-panel counts halve for wide
arms. TMA1 adds32.1/63.3us net over PACK;MC2/4 add further time despite request
reductions. These comparisons do not isolate individual waits/transport/issue.

The earlier +39/+100us oracle combined byte requests,request granularity,address
issue and cache reuse. This new receipt does not support assigning that increment
as a free activation-byte credit. It closes these exact barrier/grid/register
tradeoffs,not the activation lever. Nonblocking remote-empty multicast handshakes
and wider output schedules retaining useful parallelism require a new model/proof.

Whole-block on-chip W13→W2 is **size-rejected,unmeasured**:N32 BF16 SwiGLU alone256KiB
exceeds the115712B practical two-CTA budget. N8 rescue fits64KiB but local model
needs1006.8/1736.4MB weights and only35.6/61.3 useful CTAs on132SMs;even its4.8TB/s
weight-only floor209.8/361.8us exceeds S18 whole time. Partial-K fusion is not closed.
Gather-only coalescing premise is false,as existing sectors are full;PACK is still
tested as the necessary conversion control for multicast.

No point winners at either M;`branches/actreuse/P4_HANDOFF.json` has no nominations.
All12 exact normal/forced whole-layer screens pass13TASK/12graphs/38static;
966changed-data replay checks over69cases are bit-equalS18. Builders reproduce
root-applicable packed S18 patches with checked expanded hashes;no losing patch
is retained as a fallback. Target132.6/158.5us remains unmet.

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

Ratios are candidate/control time; SE is across paired seeds. Every measured
per-shape point ratio below1 is handed to P4, including estimates inside A:A
spread. [Complete point handoff](../problems/muse_attention_decode/dev/sliding_round2/point_handoff.json)
### P3b activation-mechanism oracle (2026-10-26)

`3330edc297f044f0`:one cheap MoE lab-only import,16 fresh seeds x3fixtures/M,
native60.0s. All9 math arms pass TASK before/after,maxL2/row .004138/.004756.
A:A signed/absolute/max seed-mean us−.043/.542/1.547(M128),−.274/.571/3.611(M256).
Packet checksum/seed archive retained in `branches/mechanism/results.json`.

S18 atomic routing changes slot order across calls. All arms therefore use one
current-fixture route outside timing,identical metadata copied to each. PRE also
moves current-input gathering outside timing and is never a candidate. BASE
176.426/231.016us is core+finish,not full MoE. Native resources confirm2CTAs;
PACK3/EARLY3 both Q3/X4,93184B,128/123regs;DUP pair both102400B+1024B,128/126regs.

| Oracle contrast | M128 ratio [95% CI] | M256 ratio [95% CI] | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|---|
|PREPACK/PACK|0.919125 [0.913731,0.924518]|0.938771 [0.932476,0.945067]|Remove current-fixture gather;about15us saved. Free gather is only .995668/.991693 of BASE.|Production zero-cost gathering is not implemented.|`3330edc297f044f0`|
|EARLY3/PACK3|1.001377 [0.999731,1.003024]|1.003610 [1.001865,1.005354]|X issued one KT128 earlier,same bytes/grid/indexing/footprint. No latency-hiding gain.|Other pipeline schedules;not a general latency bound.|`3330edc297f044f0`|
|DUPCTA/DUPSYNC|0.884004 [0.856472,0.911537]|0.816633 [0.794178,0.839088]|Same private unicast copies and2CTA cluster grid;local instead of cluster sync. Large recovery of added overhead.|Correct multicast still requires peer readiness/retirement;BASE never paid this cost.|`3330edc297f044f0`|
|MC2/DUPSYNC|0.999803 [0.997956,1.001651]|0.998526 [0.995800,1.001252]|Same sync/grid/math;multicast halves L2-side requests with no material net gain.|Byte benefit cancelled by delivery costs cannot be excluded;no HBM counter.|`3330edc297f044f0`|
|PREMC2/MC2|0.933568 [0.928091,0.939045]|0.952658 [0.946219,0.959097]|Remove gather from same multicast path;14us-scale recovery still leavesPREMC2 1.159/1.244 of BASE.|PRE is an excluded-work oracle,not a candidate.|`3330edc297f044f0`|

Literal MC2 with only__syncthreads is incorrect:one CTA cannot prove peer full-barrier
arming or prior-reader retirement. It was not sent. DUP own-destination controls
are correct and distinguish synchronization from the net multicast/request effect.
The prior +39/+100us X−W term changed8KiB activation versus36KiB weight work-item
granularity,loop/address issue and cache order;it was mixed attribution,not a
bandwidth tax. Measured new-path costs are gather and synchronization;one-stage
X lookahead gives no support. No universal zero-byte-cost claim follows.

All timed-gather arms lose BASE;PRE arms cannot promote. Free-gatherPACK .995668
[.992942,.998394]/.991693[.985817,.997570] yields only+0.21% six-load even if MoE
were100% of both verify layers;optimistic CI endpoint size illustration+0.36%,not
joint confidence or measured layer R. A >1% six-load case needs at least2.94%
reduction at each of the two affected loads even at that maximal stage share.
Combining gather-free MC2 with measured private-copy sync removal predicts still
1.024/1.016 versus BASE;interaction-dependent model,not an MC2-local measurement.

No new build or P4 nomination. Propose parking P3b by current measured decision
value pending fresh re-rank/new mechanism;the theoretical31% gap ceiling is not
closed. Handoff and every seed/CI/control: `branches/mechanism/P4_HANDOFF.json`.

records all42 winning arm cells (duplicates/controls included), one incremental
history nomination, actual seed IDs and all controls. Stage-only/older-base
results retain their scope; missing replay correctness or a failed experimental
control is unmeasured. P4 composes and applies the held-out six-load lower-CI
score gate in GOAL; no per-load significance or extra correctness gate applies.

| Recipe | Ratio vs control / A:A | Mechanism established | Leaves open (and what it closes) | Receipt |
|---|---|---|---|---|
| Decode1 full one-head N128 persistent gather, stages3/4; eight seeds | H3/H4/GC1.4240/1.3879/1.5046; A:A0.99029, SE0.00174; BF16/partial errors0.0047-0.0059 | Instrumented producer issue1714, softmax907, publication695 cycles/tile | Closes these stage schedules, not one-head full attention; no point nominee | `58d6a59603564014` |
| Sliding LAST/MASK/FAST stage sweep, six shapes, eight seeds | LAST d1/16/32 0.9307/1.0523/1.0718, v1/16/32 0.9030/0.8618/0.7997; A:A0.9856/0.9890/0.9884/0.9968/0.9923/0.9894. FAST verify1.0503/1.4051/1.2840; eight stage/partial gates pass | Padded MASK is bitwise equal; FAST changes RoPE; LAST still serializes final reduction | Closes tested FAST verify/LAST d16/d32 stage recipes. Point-winning stage pieces need full-layer TASK/score confirmation; DSM, QT and resident rebuilds remain open | `2f8b702dc76b42fd` |
| Global shared-merge last-CTA, frozen S11, sixteen seeds | **Whole-layer replay unmeasured.** Raw C128 d1S/d16S/d32S/d1F0.98987/0.99595/1.00529/1.07796; A:A1.00514/0.98012/1.00524/1.01041, SE0.00475/0.00359/0.00210/0.00298 | Isolated full-QKV attention/store gate only; captured arms check finiteness, not public TASK | No measured nomination or closure. Repaired replay TASK/cache cohort below; route differences alone are not a correctness rejection | `ae6cd27877cc4ee2` |
| Native verify sliding DSM C1/2/4/8/16, eight seeds | v1 C4/C8/C16 0.947485/0.773111/0.719269, SE0.003430/0.002597/0.002772; v16 C4/C8 0.966488/0.849109, SE0.003992/0.003406. v32 bestC4 1.07381. A:A0.96012/0.97988/0.99157, SE0.00336/0.00328/0.00309; all stage/partial gates pass | Cluster CTAs distribute output-row reduction; no isolated phase attribution | Hand all five stage point winners to P4, adding the smaller C4/C8 signals previously omitted. Selected whole-layer follow-up completed below; v32 tested sweep loses, other schedules remain open | `b05713ecf6574ad9` |
| Compact early decode1 history E4/E2/E1/lateL4 vs exact OLD4, exact S14;16 seeds x12 rounds | E4/OLD4 **0.997414**, SE0.001244; E2/E1/L4 vsOLD4 1.016159/1.040193/1.020247. OLD4/E4/E2/L4 vsA0.978186/0.975761/0.994125/0.997971, SE0.014096/0.014496/0.014870/0.014298. A:A1.009555, SE0.006105, range0.98824-1.09546; all TASK gates pass | QT4 transpose median/max4.064/4.608us precedes wait5.152us; shared53504→28928B. E4 attention increment12.160 vsOLD4 11.200us; total attribution not isolated | **Verdict changes: reopen E4 and nominate all four vsA point winners to P4.** Withdraw A:A/3SE rejection. P4 confirms current-base composition; no separate history send required by the old rule | `402819c28279466d` |
| Selected DSM whole-layer cohort, exact S14;24 held-out seeds x12 rounds | **v1 unmeasured failed control:** RESET/COOP/HALF/DOUBLE0.96309/0.94983/0.95830/0.98687, A:A1.00963, SE0.00201; identical C16 DOUBLE/COOP1.03901, SE0.00300. v16 RESET1.00412, SE0.00395; cooperative C8/C4/C16 1.11715/1.06338/1.15331; A:A0.999657, SE0.000744. All TASK/cache gates pass | v16 cooperative QT1 uses254 registers/2 active CTAs vsRESET168/3; attention51.648 vsA24.128us. Consistent with register pressure, not isolated causation | Unchanged: no v1 estimate from this failed experimental control. Tested v16 QT1/gang recipes lose. Fresh-pool repeat below resolves v1; different verify schedules remain open | `6c26b14fdd28458b` |
| Verify1 DSM fresh-pool repeat, exact S14;35 disjoint seeds x14 rounds | RESET/COOP/HALF/DOUBLE0.963624/0.951214/0.966865/0.954611, SE0.005857/0.005155/0.004950/0.005723; pooledC16 **0.952913**, SE0.005071. A2/A3 0.999262/0.999499, SE0.004232/0.003422; duplicate/COOP1.003740, SE0.004151; all TASK/cache gates pass | Fresh graph pools/rotated ownership remove the prior cohort's duplicate discrepancy; seed cohort also changed. First attention increment17.632→12.064us, QKV-0.032/O+0.352us | Hand every point-winning arm to P4, retaining QT4/C16 priority. Candidate403bb778 normal/forced each13 TASK/12 graphs/24 static, explicit local DSM C1. Current-base composition and six-load confirmation are P4's decision | `6a0b3f0987e94ba5` |
| Round2 last-CTA/QT/early/DSM, exact S14;16 seeds x12 rounds | d16 D8 **0.995283**, SE0.007209; d32 D4 **0.989822**, SE0.007613, L4 **0.993761**, SE0.007368; A:A1.007797/1.002143, SE0.001748/0.000707. Seven v1 point winners: L8Q8/L16Q8/E8/D8/D16/DQ8/RESET_D16 0.950031/0.962136/0.971743/0.983102/0.924229/0.950443/0.930062; A:A1.009469, SE0.002417; all TASK gates pass | v1 QT1 attention76.608us vsQT8/split8 12.864us andD16 11.840us; D16 shader/base match35-seed repeat | **Verdict changes: nominate d16 D8, d32 D4/L4 and all seven v1 point winners.** Withdraw per-load A:A/3SE exclusions. Decode selectionc2000b3c passes normal/forced13 TASK/12 graphs/24 static each (DSM C1 locally); handoff.json rebuilds it. P4 composes/confirms. Losing measured tilings close only themselves | `27a914f72bfb47c8` |
| Resident verify history, C2/4/8/16 pipeline, C4/8 serial, RAW4/8;24 seeds x12 rounds | Bestv16 C8S1.012806, SE0.003340; v32 C4S1.020146, SE0.002518; A:A0.999296/0.998184, SE0.000342/0.000252. All10 arms/shape pass TASK/cache; no point winner | v32 C4P starts2.847us before QKV end20.063us (RAW4 20.128). QKV221508B leaves11964B/SM, below smallest attention27648B. v16 C16 fits and starts4.513us before QKV13.792us but loses1.05542 | Unchanged: close only these resident/serial/edge schedules, no P4 nominee or follow-up. Lower-footprint producer/different gather/transpose/output schedules remain open | `857f78fa889f4646` |
| Repaired global pair, exact S11 shaders;16 seeds x8 rounds, all replay TASK/cache gates pass | Bestsliding d1 C1280.974683, SE0.002396; d16 LAST0.975854, SE0.007705; d32 PP0.986235, SE0.002162. A:A1.003476/1.000554/0.999184, SE0.001814/0.002448/0.001713. Full d1 globalC128/C64/32split/64split1.07712/1.15091/1.25743/1.10016; A:A1.000601, SE0.002014 | Full C128 attention33.024us vsA24.640/PP22.688: one launch leaves extra critical work. Default/fixed-split and OLD/PP controls agree | **Verdict broadens: hand all14 point-winning cells to P4 with S11 scope**, including smaller signals and old controls. Full global recipes still lose. Exact-S14 follow-up58f0b30b is collected below | `bb7239afc1b3487a` |
| Current-S14 global128/identicalALIAS/exactLAST/PP at decode1/16/32 sliding;35 fresh seeds x14 rounds, recaptured pools | d1 GLOBAL/ALIAS/LAST/PP1.006456/1.006122/1.012486/1.001040; d16 **0.981291/0.978600/0.978764/0.992226**, SE0.004735/0.004761/0.005667/0.002863; d32 1.000403/1.000920/**0.999170**/1.000245, LAST SE0.004589. A2/A3 d1 1.001140/0.999445, d16 0.998170/1.001622, d32 0.999711/0.999883; A2 SE0.002019/0.002669/0.000457. Every7-arm/35-seed TASK/cache count passes | ALIAS/GLOBAL0.999689/0.997382/1.000510 at d1/16/32, SE0.001057/0.002762/0.000587: identical controls agree. d16 global and last one-launch paths have lower point estimates; cause is not isolated | Nominate all four d16 cells and d32 LAST to P4; d1 has no point winner in this cohort. No universal-family rejection from a small noisy loss. Source/payloadb70ff21c, wrapperb899a8fb, sender1249027; no P5 receipt pending | `58f0b30ba8b8472f` |
| S15 sliding K-ready/V-ready shared-full review audit; exact60254883 source and native dispatch | No new timing arm: reviewed barrier absent from sliding. Source-level evidence in dev/operand_ready/audit.json | d1 sliding_main3815beab and d16/32 attn_mainb63e42f6 have no full ring. Verify sliding dispatches FA3; VQ8 rows are only full16/32. VQ8 producer transpose→full arrival/consumer next-QK wait is real only on full | Drop the specific sliding shared-full/VQ8 claims, build/send zero no-effect arms. Full VQ8 split is P5b's item. Serial decode/FA3 rescheduling remains open; no new speedup claimed | `7eaa262a30ec483e` + S15 source |
| Exact-S15 FA3 sliding diagnostic: B1/8/16/32, windows512..8192, requested splits0/1/2/4/8 plusA2;3 seeds x6 cold rounds | At2048: force1/default v16 **1.267437**, SE0.007414; v32 **1.085305**, SE0.004072. Force8 v16 1.015415, SE0.016892; v32 **0.994060**, SE0.005777. A:A1.011205/0.997109, SE0.003422/0.003807. Every arm/seed public attention TASK passes; all ratios/seed IDs in dev/window_rebuild/diagnostic_analysis.json | Full symbols: five launches; prepare_varlen_num_blocks is metadata, not V transpose. Main14.144/19.392us, combine4.064/4.416us; default kernel span22.464/28.256us. Window slope implies1.808/1.855TB/s effective KV rate, including math/SFU/scheduling, not isolated DRAM saturation. Exact raw S15 critical22.923/28.490us (table17.01/21.14 is46/62 weighted) | Closes forcing one split at2048; split tuning does not close1.3KV targets2.27968/4.55936us. The weak n3 v32 requested8 signal was sent to packed current-base confirmation; n24 af563b6b below changes it to a loss. No stage-only promotion; other windows remain open. One-launch resident/streaming DSM arms address launch and mainloop costs | `45ab5a9beecd407b` |
| Exact-S15 two-WG resident/streaming DSM verify sliding; A/A2/F8/WG2C2/WG2C4/STREAM2/STREAM4/RAW2,24 seeds x10 cold rounds;384 public layer TASK gates pass | v16 H2/H4/S2/S4/RAW2 1.018941/1.071535/1.002720/1.036232/1.004190; no point winner. v32 1.030890/1.080038/**0.997895**/1.045406/**0.982930**, S2/RAW2 SE0.002301/0.001788. F8v32 1.002189 SE0.001145; v16 F8 unchanged. A:A v16 **1.001235 SE0.000711**, v32 **1.000291 SE0.000385**; seed IDs/all arm SE in dev/window_rebuild/pair_result.json | Default actual FA3 splits4/2, requested8 gives4-5/2-3. One launch, two independent KV-owning WGs per CTA, FP8 Q/PV, deferred new-KV prep and DSM merge. v32 first-capture attention A28.992us/earlyS2 22.144/RAW2 24.160; bulk early overlap not established (94208B vs11964B producer spare). C2/C4 admit66/30 clusters, so64 head groups need1/3 waves. Clocked S2 remaining math/new-KV9.792us, merge/gate3.392us; instrumentation changes registers | **Nominate both v32 RAW2 and earlyS2 to P4 S18, RAW2 preferred; retain FA3 v16.** Exact v32-only f47ece74 passes normal/forced13TASK/12graphs/25static each; tracked P4_HANDOFF.json/P4_raw32.patch. Requested8 prior signal does not confirm. Closes only tested losing resident/C4/v16 recipes; targets2.280/4.559us remain unmet. PIPE2 math-schedule follow-up47e1f9d82ee745e9 (sender2882877),18 new seeds, collected below after10component/16forced-payload gates and native no-spill compile | `af563b6bc1af490f` |
| Exact-S15 follow-up A/A2/RAW2/PIPE2;18 new seeds x10 cold rounds;144 whole-layer TASK gates pass | v16 RAW2/S15 **1.008379 SE0.003318**, PIPE2/S15 **1.023038 SE0.003421**; v32 RAW2/S15 **0.975938 SE0.002329**, PIPE2/S15 **0.982128 SE0.002075**. PIPE2/RAW2 **1.014540 SE0.000893** at v16 and **1.006358 SE0.000661** at v32. A:A **0.999917 SE0.000853 /0.999382 SE0.000920**; all seed IDs in dev/window_rebuild/pipeline_result.json | Same C2 streaming DSM and ordinary edge; only next-QK/current-PV/softmax schedule changes. Native regs172→255, zero spills; instrumented v32 math/new-KV9.856→10.752us, publication2.432→4.176us, merge/gate3.424→3.872us. First-capture attention S15/RAW2/PIPE2 29.472/24.033/25.344us. More live operands/barriers and longer phases are observed; sole cause not isolated | **Closes PIPE2 as an improvement over RAW2 at these shapes.** Still hand v32 PIPE2 point nominee to P4 S18, exact61224be8 patch normal/forced13TASK/12graphs/25static each; RAW2 remains preferred and repeats its win. No v16 nominee. Target4.559us unmet, coordinate/transport attribution and smaller shared footprint/different producer remain open; no further P5 job pending | `47e1f9d82ee745e9` |
| Sliding mechanism oracle, native D8/D4 decode and RAW2 verify plus exact S15 FA3;12 fresh seeds x8 cold,316 public TASK gates | Order d16/d32/v16/v32: precomputed-offset C/M **1.007013/1.013572/0.992287/1.003528**; constant-softmax S/C **0.932858/0.892509/0.814772/0.827377**; full/no-merge N/M **0.844421/0.883146/0.830538/0.837853**. O2/O **0.992690/0.996913/0.994085/0.994747**, M2/M **0.996239/0.994544/1.001899/0.997449**; directional0.5–0.7pct control drift retained. All paired SE/95pct CIs and seeds in dev/sliding_oracle/oracle_result.json | Same loaded M binary for all ablations; M/O overhead1.0160/1.0433/1.0201/1.0236. Order-dependent exposed costs us: launch2.17/2.20/2.06/2.21; setup/address-copy issue4.56/6.09/7.27/7.33; coordinates-.12/-.30/+.23/-.11; KV transport without math1.90/4.49/2.43/4.59; softmax/packing1.17/2.45/5.37/5.32; QK/PV/transpose/newKV1.91/2.40/3.97/3.57; constant-P merge2.69/2.22/4.91/4.82. Real merge removal2.68/2.63/4.95/4.98us. PDL inputs are ready; active producer wait is not inferred. Verify first-copy/publication stamps absent, repaired in new context probes | Largest verify exposed costs are softmax, setup and merge. No useful large coordinate gain; weak deltas remain unresolved under control drift/extra64bit coordinate reads, not a general hoisting rejection. Build one valid EXP probability block (FP32 affine exponent construction, online max/rescale/sums and FP8 PV retained),24 fresh/strided TASK plus normal/forced13TASK12graphs25static; native per-key SFU sites fall. Sent whole-layer current-P5-base/identical/RAW/EXP7bd3eced41e24402 (sender3610930),16newseeds, after exact40layer+8real-producer attention TASK gates; no altered-work oracle nominates | `f365764142034130` |
| SFU probability EXP response to f365 oracle, exact S15 other stages;16 fresh seeds x10 cold,320 layer+8 real-producer attention TASK gates | EXP/current-P5-base d16 **0.996181[0.984994,1.007368] SE0.005249**, d32 **0.998138[0.987856,1.008420] SE0.004824**, v16 **1.013426[1.004386,1.022467]**, v32 **1.008693[1.003643,1.013742]**. A:A **0.988514/0.998694/0.999121/1.000771**. D16 identical RAW/INC0.991353, EXP/RAW1.005055; its point lies inside directional duplicate drift | Per-key FP32 exponent-bit construction replaces SFU exp, preserving online maxima/exact rescale/FP32 denominator/FP8 PV/every key. Native SFU91→23decode,78→10verify; FMA/F2I issue grows. Instrumented probability cycles2374→3086,4605→5962,8444→10575,8308→10322. Real selected-projection median PDL waits64/64/240/256ns; d16 qkv3 has10FP32partials,d32oneFP32,v16threeFP32,v32BF16. Whole-layer critical attention v32 23.904→24.800us | **Revise SFU-unit hypothesis:** constant-softmax isolated max/exp/sum/packing together; fewer exp instructions did not accelerate this block. Close affine exponent recipe at verify; retain FA3v16/RAW2v32. Hand both decode point estimates to P4 S19 with control drift/SE/seeds and exact decode-only25325c56 patch (normal/forced13TASK12graphs29static each); no shader or G0 gain established. Next test the other5us term: same RAW2 math/slices with nonspinning device-memory last-WG merge and explicit acquire/release publication; native+12component TASK, normal/forced13TASK12graphs25static and packed20layer+4producer TASK pass; sent attention check3a9a31747042429a,sender4068097,16newseeds10cold | `7bd3eced41e24402` |
| RAW2 math/4 logical slices, one-launch nonspinning global last-WG reduction;16 new seeds x10 cold,160 layer+4 real-producer attention TASK gates | GLOBAL/currentINC v16 **1.087312[1.081501,1.093122] SE0.002726**, v32 **1.051840[1.049692,1.053988] SE0.001008**. GLOBAL/RAW **1.077219[1.073683,1.080754] /1.051936[1.049579,1.054294]**; A:A **0.999772[0.996889,1.002655] /0.999299[0.998218,1.000379]**. All seeds/CI in dev/sliding_oracle/merge_result.json | Same172regs/94208B and QGMMA9/LDGSTS24/SFU78; explicit acq_rel publication/cg reads/release reset. Unconstrained single-CTA slots132 vs66 two-CTA clusters, still1CTA/SM. Last WG handles64 rows at16rows/warp versus four WGs at4rows/warp. Clocked publication→finish21.072/16.672us versusparallel4.544/3.392us; work per role differs. Main probability cycles remain similar. First-capture critical attention INC25.601/24.065→GLOBAL44.064/38.656us | Close this last-WG global recipe, not last-CTA or global reduction generally. Lost merge parallelism plus partial traffic/fences dominate; preserve FA3v16/RAW2v32. No GLOBAL/verify EXP nominee; two uncertain decode EXP points/CI/seeds/patches are in P4_HANDOFF.json. Next must retain merge parallelism and cut max/sum/packing or setup/issue. Targets2.280/4.559us remain unmet; all three P5 receipts collected, none pending | `3a9a31747042429a` |


| S19 residual audit and finer same-binary setup oracles;12 fresh seeds x8 cold,96 full-work oracle TASK | M2/M d16/d32/v16/v32 .992023/.992579/.996870/.991583; directional0.3–0.8pct drift retained, weak partitions unresolved. Z-E6.987/10.954/10.668/10.654us is from a new binary, not the earlier4.6/6.1/7.3/7.3us residual. All SE/CI/seeds in dev/setup_tree/result.json | Query work is BF16 or FP32-partial load/sum→BF16, RMS/warp reduction, RoPE, E4M3 pack/shared layout; decode also newKV/gate. Metadata RoPE and first2prefetches already precede PDL. No TMA; LDGSTS24sites, ordinary page LDGs; WGMMA descriptors are integer arithmetic. Ordered BF16-ready removals us: values.290/.289/2.221/2.116, sincos.383/.500/.233/.285, remainingprep1.441/1.394/1.296/1.204, zero-copyissue2.738/3.664/1.057/1.169, offset/address/loop1.059/4.017/4.967/4.941, protocol1.075/1.090/.893/.938. Exact S19 device-body/SASS audit retained | Withdraw interpreting the old residual as Q alone or descriptor setup. New volatile address guard/branches/register allocation prevent overlaying the two splits. No altered-work arm nominates; drift is uncertainty, not a hardware attribution. Actual producer-partial cost is not measured by the ready-BF16 oracle. Test physical-page hoisting on valid S19 pairs | `b1a50bae4feb4ad1` |
| S19 one-Q-owner per cluster plus DSM Q broadcast;12 seeds x8 cold,216 whole-layer TASK across Q/tree/control cohort | Q/S19 d16 1.024033[1.020005,1.028061],d32 1.008590[1.005388,1.011792],v16 1.018728[1.006200,1.031256],v32 1.003218[.999975,1.006461]. Q/RAW1.021559/1.010555/1.003671/1.004207; A:A.996862/.998634/.999980/.997772, every95pctCI contains1 | Query loads really predicated off at nonowners; one split normalizes/ropes/packs, then vector DSM copies publish Q to peers. KV/newKV/gate/math/merge unchanged. Native regs138/172 unchanged, no spills; adds startup cluster barrier and Q copy. Sole regression cause not isolated by clocks | Closes this Q-broadcast recipe at all4loads, not producer-epilogue fusion. No P4 point. Normal/forced13TASK12graphs6static each plus native compile; query-owner source/math execute in local C1/two-WG stand-ins. No P10 producer changes | `b1a50bae4feb4ad1` |
| S19 verify parallel DSM tree:8warps/CTA own disjoint rows for local pair reduction, every WG shares final two-CTA DSM merge;12seeds | T/S19 v16 1.018511[1.006539,1.030483],v32 1.001648[.999438,1.003858]; T/RAW1.003469[1.000928,1.006010] /1.002648[.999441,1.005855]. A:A.999980/.997772, CIs contain1 | Keeps reduction parallel; unlike prior last-WG, no64row serial owner. S19 v32 already has fused parallel DSM gather, so tree adds local partial stores/reduction/barrier; native regs172→197, same CTA occupancy, no spills. v16 original RAW2/S19 loses1.014970 | Closes this two-level tree, not DSM generally. No point nominee; retain FA3v16/RAW2v32. New valid H arm preloads one physical-page map/key into registers before PDL and quad-shuffles coordinates to copy sectors.16component+32packedTASK, normal/forced13TASK12graphs each, native140/168regs no spills; sent prefill import6b15d130e20f439f(sender1466444),16newseeds10cold after attention quota spent and unsent waiter1447395 cancelled; result below | `b1a50bae4feb4ad1` |

| S19 H cooperative register page map before PDL;16 new seeds x10 cold,256 layer TASK; prefill-host import after attention quota spent | H/S19 d16 1.005026[1.002175,1.007876],d32 **.999373[.997703,1.001043] SE.000784**,v16 1.001271[.994404,1.008138],v32 1.000132[.998087,1.002178]. H/RAW1.000529/1.000746/**.997187[.995800,.998574]**/1.000536. A:A.997644/.998277/1.000299/.999751, all95pctCIs contain1 | Four quad lanes own four tiles; physical-page/row map loaded once/key into registers before PDL, quad shuffles supply copy sectors. Native LDG.E sites41→33 /35→27, four page LDGs in prelude and none in source-attributed copy loop. Regs138→140 /172→168, no new shared/spills, same occupancy.144 CPU cases/204800 sectors prove old/new coordinates, TASK covers fresh/partial/page64. Native v16 gains0.28pct vs RAW but remains above FA3; no native5us coordinate term recovered | **Hand only d32 H changed-recipe point to P4 S19**, with exact557deae0 selected shipment/patch, normal+forced13TASK12graphs6static each, all SE/seeds/control alternatives. D32 H/RAW1.000746 and A:A.998277 mean no isolated shader win. Preserve FA3v16/RAW2v32 and unselected shapes. Closes large-gain expectation for this coordinate recipe, not hoisting generally; producer fusion, copy/transpose and max/sum/packing remain. No standalone small-gain repeat; both receipts collected,568 native TASK total | `6b15d130e20f439f` |


### P5 S19 producer-finish sizing (2026-10-26)

The supplied2.115/2.183/3.750/3.605us preparation removals are old RAW2
BF16-ready altered-work estimates, not actual S19 FP32 reduction/FA3 costs;
S19v16 uses FA3. P10's logical bytes/time and static SASS cannot establish
idle ALUs. Exact coded d1/v1/d32 output tiles own16of64head columns, and
wait for K partials before output. A **global last-CTA finish starts after
the weight stream**, so it cannot hide under that same stream. Head-local
readiness could overlap other heads but needs another ownership/schedule;
v32 uses cuBLASLt. No production fused patch or candidate ratio is measured.

| Recipe | Ratio vs control | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Initial producer-finish sizing,8seeds/all12S/Fshapes | Whole-layer baseline comparison **unmeasured**: A profiled versus A2 plain capture;192layer+96finish TASK pass. d16/v1 A:A .99355/.99537 are retained, not treated as a speedup | Unequal capture procedures confound A:A; repaired in next receipt with identical plain captures and separate profiling | No closure/promotion from baseline timings; immutable source and invalidated analysis retained | `95b406f48c8e45c7` |
| Repaired S19 producer+standalone ready-Q/KV finish versus producer+same-resource empty;8freshseeds,6cold | Weighted46S+16F **component PF/PE** d1/d16/d32/v1/v16/v32 **1.347849[1.329518,1.366179]/1.581056[1.569378,1.592735]/1.257118[1.243818,1.270419]/1.258842[1.245398,1.272287]/1.285815[1.273549,1.298082]/1.106339[1.099603,1.113075]**. Layer A:A1.004944/1.000923/1.001441/1.002432/1.001374/1.001868;192layer+96ready-output public TASK pass | Real FP32/BF16 ABI, reduce→BF16→RMS/RoPE→E4M3 Q/newKV;40regs/no spills. Paid S tails4.124/8.017/4.209/3.745/5.147/2.761us. v32 A:A+0.187% directional drift retained, no per-load veto | Standalone finish is not a lower bound for register-resident fusion or a candidate/S19 ratio. No new P4 nominee. Global free-tail premise fails, head-local fusion remains unmeasured | `116dd8976c844b54` |
| S19 sliding completion-critical increments versus1.3KV byte targets, same eight seeds | Critical us7.676/11.836/16.492/12.392/22.832/24.028; targets.142/2.272/4.544/.142/2.280/4.559; **excess7.534/9.564/11.948/12.250/20.553/19.469us** | Separate CUPTI calibration per fresh fixture, not summed overlapping phases; all paired CIs/SEs/seeds retained | Profiling perturbation, small-batch latency and math/issue prevent calling all excess recoverable; large theoretical attention gap remains | `116dd8976c844b54` |

The four supplied RAW2 deletion cells, d1/v1 unchanged, modelR1.005628 for
46sliding layers or1.007604 if extended to62, at zero added producer/ready
load cost. These are scenarios, **not six-load upper bounds or promotion**.
The parking proposal based on them is withdrawn: it did not size the lever.
The requested consumer-side deletion oracle below replaces that rationale.

### P5 consumer finish-deletion oracle on S19

R is BASE/DEL, larger is faster. Eight fresh seeds85000121+104729*i,
six complete palindromic cold rounds; all1728 round times and384 device
telemetry reads retained losslessly. Exact S19 62ea5bf4, all six S/F loads,
192 BASE/A:A public TASK+96 prepared-fixture TASK pass. DEL deliberately
prepares ready current Q/K/V/gate and FA3 metadata outside timing; it is
never a candidate, even with correct frozen-fixture numerics. Producer,
gate GEMV, all-key attention, merge and downstream work remain timed.

| Recipe | Ratio vs control | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Initial consumer deletion oracle | **Unmeasured**: missing required identities/seeds packet fields; RAW2 DEL cleared live history,8/8 v32S fixture failures. BASE/AA192 and otherDEL88 pass; all samples/source retained | Wrapper flags incomplete evidence; same-fixture sanity detects native implementation fault | No timing verdict/closure. Required schema and RAW2 storage lifetime fixed, exact wrapper/schema/tail/codec rehearsed, new seeds used | `510d394fdc604eba` |
| All-path DEL, exact-S19 whole-layer oracle | **Six-load R1.015900[1.010938,1.020887]**, paired-logSE.002071,AASE.001800. d1/d16/d32/v1/v16/v32 **1.025467[.998939,1.052700]/1.009522[1.006028,1.013028]/1.004511[1.001571,1.007460]/1.032477[1.012810,1.052527]/1.014188[1.010295,1.018097]/1.009520[1.001272,1.017836]**. Six-load A:A1.000336[.996087,1.004602] | Delete native head prologues/QKV wait, full-decode sumcast and FA3 prep+metadata; packed ready transport remains. Six source-boundary proofs, original partitions pinned, native execution and all prepared-fixture TASK pass; wire8162B/completeness/checksum valid | **Park consumer finish family per prescribed LB1.015 decision**: actualLB1.010938. No build/P4 nominee. This does not close all sliding or producer mechanisms. Most native paths already fused the finish, so standalone paid tails were not additive deletion costs | `ca022370c1f34adc` |
| Sliding-only /full-only derived deletion, same seed pairs | Six-load **DEL_S1.011513[1.006474,1.016578] /DEL_F1.004287[1.000021,1.008570]** | Combine measured changed path with same fixture's unchanged BASE path; no separately timed duplicate calls | Both optimistic exclusions, never correct nominees or additive stage timings. Prior d32 page-map point remains with P4 | `ca022370c1f34adc` |

Controls retain d1's wide CI and v1 A:A+0.967% directional drift, with no
per-load veto, trimming or selective rerun. SM1980MHz/memory3201MHz fixed,
32–39C; worst archived round/median1.09869. No clock counter identifies a
cause for the remaining variation. Native counts confirm full d16/d32
sumcast removed (10→9/11→10), FA3v1F/v16S prep+metadata removed (13→11/20→18),
math/merge retained. **v1S is native R2 DSM QT4/C16**, not FA3, by source,
actual selections and nodes. Most native paths already reduce partials,
roundBF16, RMS/RoPE and pack within head-owning attention prologues.
Half-deletion point illustrates only~0.79% six-load, not a measured recovery
or a universal bound. [All CIs/seeds/controls and evidence-only P4 handoff](../problems/muse_attention_decode/dev/consumer_finish/README.md).


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
| Small-M/generic V9 entry release in grouped E | E/no-MoE 1.007300/.999919/.998656/1.001767 at d1/d16/d32/v1 | Changed native code compiled/executed; outputs agree | Point gains at d16S/d32S/d32F are nominated under the revised rule; exact held-out composition remains P4 | `f88fda4aac144374` |
| S15L all library replacements | .999802/.999176/1.001312/1.006730/.994152/1.005131 vs S15; all768 TASK gates pass | Native NF O, VDSK pool/unpool and VQ8 rows replace external completion boundaries | Full composition is not a winner; isolate candidates below, do not blame all libraries or all native kernels | `f22020d25d234be6` |
| Owned NF O128/O256 | L/no-O .995180/.998636 at v16/v32; A:A .999198/.999509 | Same BF16 inputs, owned release/wait chain; exact native kernels used | Both O128 and O256 point gains are nominated with uncertainty; held-out C composition below confirms O128 with the tested release chain | `f22020d25d234be6` |
| Owned VDSK pool128 S3,128×128×64,st5/w8,TMA | L/no-pool128 .998126 at v16 | FP32 partials feed original route with PDL release/wait | Small tentative signal; confirm pool alone and beside O128/PDL | `f22020d25d234be6` |
| Owned VDSK unpool128 S2 | L/no-unpool128 1.003428 at v16 | FP32 partials feed original boundary2; TASK passes | Close this tile/split recipe as a win; other owned unpool designs remain open | `f22020d25d234be6` |
| Native N128 VQ8 st4/mt2 at v1 full | L/no-v1full 1.007972 at v1 | Every full key/current token retained; TASK passes | Close this FA3 replacement recipe in the full layer; alternative native designs remain open | `f22020d25d234be6` |
| Native N128 VQ8 st4/mt2 at v32 sliding | L/no-v32slide 1.006421 at v32 | Every window key retained; TASK passes; increased seed spread | Drop this replacement; no claim that all native sliding loses | `f22020d25d234be6` |
| S15 vs range-dispatch S15n vs SGLang,65 shapes | Nearby/scored speedup by d1/d16/d32/v1/v16/v32: S15 .9547/.8651/.8184/.8119/.7327/.7263; S15n1.0292/1.0142/.8884/.8682/1.0050/.9360 | Complete2seeds×8coldreplays, SGLang available, finite/schema gates pass; many unknown-B calls otherwise lose specialized MoE/attention/projection/glue. Same scored dispatch control ranges .97794–1.03416/.99554–1.00231/.99239–.99919/.99428–1.00065/.99491–1.00196/1.00054–1.00283; no independent A:A and slight d32/v32 drift | Large nearby cliffs improve; no scored gain claim or correctness promotion. Native TASK67-shape validation passed268/268 checks as c03a3ae8ddfd490d; guarded S15n9cc90d2e is handed to P4 as a whole and B<=64 preserves all65 timed cases. Other neighboring contexts/batches and final integration remain open | `a793c99eb9d447df` |
| P5b point-estimate rejudgment of S15e/S15L |19 affected shape cells have on/off point ratio<1;8seeds; frozen per-shape SE absent, full/off load SE retained in handoff | Re-evaluate cells instead of vetoing source groups by every load/A:A; exact source and root-applicable patch for each group | Whole guarded S15n, O256 and small-M release point gains now nominated too; P4's held-out six-load R is the promotion gate. Both softmax reset batches still have no changed-arm point gain | `f88fda4aac144374`, `f22020d25d234be6`, `a793c99eb9d447df` |
| PV-group denominator + vector-gather payload | **Unmeasured**: native denominator probe fails before timing, rel .0004769067 at seed68137 | Local portable exact sum did not establish native tensor partial-product accuracy; LAB_ERROR aborts the packed experiment before TASK/timing | No loss/win claim for either denominator or gather. Other denominator layouts and independent gather testing remain open | `fb14f30ffd48450f` |
| S15 PDL/library held-out composition C | R1.006161,95%CI[1.004298,1.008028];A:A R.998390[.996468,1.000315];8seeds,1440TASK pass | Preselected early q1 attention+large-M MoE+VDSK release+native O128/pool128; C_no_o R1.005248, C_no_pool1.004273, P1.004848 also clear G0 | Exact C patch/source in dev/pdl_release/confirmation/P4_HANDOFF.json promoted to P4 against S15; interaction with later candidate remains P4 confirmation, no per-load veto | `60bad72bcf964453` |
| VQ8 K-ready/V-ready with explicit PV wait | v16F/v32F1.001874(SE.000562)/1.007053(.000688);8seeds,854TASK pass; affected-shape A:A v16/32 .998796/.999073,d16/32 1.002203/.999249 | Separate K publication before V transpose; QK waits K, every PV waits V | Tested explicit-wait form loses; redundant wait removal and other overlap remain;six-load A:A R.996636[.994997,.998278] drift retained, not per-load veto | `1bab0b2d0dc14931` |
| VQ8 K/V-ready plus proven PV wait elision | v16F/v32F.997130(SE.000760)/.998618(.000929);8seeds,854TASK pass; affected-shape A:A v16/32 .998796/.999073,d16/32 1.002203/.999249 | K(next) from all4producer warps implies prior V(current); final PV retains V-ready wait | Both point winners to P4, exact ready_elide.patch7a7d9cc5; standalone R.996507[.994361,.998658] does not clear G0;six-load A:A R.996636[.994997,.998278] drift retained, not per-load veto | `1bab0b2d0dc14931` |
| VQ8 register-transfer control | v16F/v32F1.015525(SE.000933)/1.033249(.001325);8seeds,854TASK pass; affected-shape A:A v16/32 .998796/.999073,d16/32 1.002203/.999249 | Producer32/consumers232; SASS consumerR193 and producer<R32, dynamic transfer real | Tested transfer schedule loses; register/occupancy alternatives remain;six-load A:A R.996636[.994997,.998278] drift retained, not per-load veto | `1bab0b2d0dc14931` |
| VQ8 even/odd online-softmax states | v16F/v32F1.033005(SE.001895)/1.048299(.001689);parity/reg1.017210/1.014567;8seeds,854TASK pass; affected-shape A:A v16/32 .998796/.999073,d16/32 1.002203/.999249 | Two statically indexed m/l/O chains, exact final log-domain merge, same register transfer | Tested parity schedule loses beyond reg control; other overlap schedules remain;six-load A:A R.996636[.994997,.998278] drift retained, not per-load veto | `1bab0b2d0dc14931` |
| q1 K-major native FP8 M64/N8 QK | d16F/d32F1.193606(SE.006464)/1.243381(.003908);8seeds,854TASK pass; affected-shape A:A v16/32 .998796/.999073,d16/32 1.002203/.999249 | Native QGMMA64x8x32, exact3layout probes, unchanged gather/split/PV; shared score remap | Current shared-remap form loses; direct accumulator softmax/PV mappings remain;six-load A:A R.996636[.994997,.998278] drift retained, not per-load veto | `1bab0b2d0dc14931` |
| Whole guarded S15n native validation |268/268 TASK checks,67shapes×2seeds×2arms,no errors/failures | Real Hopper range kernels, all65timing shapes plus B64/B65 capacity and captured replay/cache/routing;267.69s | Whole guarded range patch792eacc3/source9cc90d2e goes to P4; scored dispatch identical, final composition uses G0 | `c03a3ae8ddfd490d` |
| K-major dead-K score reuse |d16F.999788(SE.002715),95%CI[.993369,1.006208];d32F1.007311(.004102)[.997612,1.017010];8seeds,488TASK pass | Native API occupancy A/old/reuse3/2/3 proves the6144→2048Bstatic repair removes old2CTA regression | Only d16F point to P4, patch57130acd/source4b8641bc; no standalone follow-up for this tiny point. Exact shared-remap form lacks large gain; other direct accumulator/transport paths remain | `31730357bae64c9c` |

### P5b full-attention roof-gap attribution

Full columns in the S15 stage table are weighted16/62. The raw per-full-layer
median-seed critical times below come from the retained trace, so they differ
slightly from the three-seed weighted means. These are logical byte rates,
not hardware HBM counters; 4.58TB/s is the best fitted whole-array stream, not
a universal page-row ceiling. PDL wait residence is clipped at the predecessor
frontier and is never charged twice. `dev/full_attention/roof/timeline_account.json`.

| Load | Raw critical/roof us; mechanism | Structural test | Evidence |
|---|---|---|---|
| d1F |19.36/6.90;one launch,336split CTAs,1.05TB/s logical. Fixed prologue/short-tile/merge leading hypothesis, new97d6fa2b clocks show9089/9424prologue/epilogue vs8214loop cycles,2–4tiles;pageissue341/tile | F balances whole64-key tiles; no launch is deleted because none is extra | `7eaa262a30ec483e` |
| d16F |104.64/73.87;3.17TB/s;sumcast adds1.18us. Exact q1 page/issue1578cycles/tile vsQK442,smx259,PV349;gather-only3.62TB/s | G quad-consecutive rows and vector page reads, same all-key gather/splits/math | `7eaa262a30ec483e`,`5c99583676994c2a` |
| d32F |194.53/146.09;3.40TB/s;sumcast .86us. Page/issue1672 vsQK437,smx256,PV345;gather-only3.74TB/s | G; new exact-source97d6fa2b pageissue1698 vswait25cycles/tile | `7eaa262a30ec483e`,`5c99583676994c2a` |
| v1F |29.86/7.01;prep2.40+metadata3.30+FA3main17.15+combine5.95+norm1.06uscritical;idle gap.065us | P one native partial-input launch,H1/H2 fused native FP8 path. Removes fixed kernels/work, not a claim all residence is launch cost | `7eaa262a30ec483e` |
| v16F |136.77/75.17;main130.46us2.56TB/s;combine adds6.31us of25.51residence. One384thread/221184B/168regCTA perSM. Actual S15N128softmax1633cyc/tile,producerpage/copy1280,Vpublish631,MMAissue305;waitQK46/PV9 | H1/H2 onehead256threadCTA,110592Bdynamic,128entryregs and producer32/consumer216. H2 doublesgrid to use2CTA/SM;H1 control | `7eaa262a30ec483e`,`5c99583676994c2a` |
| v32F |231.97/149.35;main227.20us2.96TB/s;combine adds4.77us of24.80residence. Actual S15N128softmax1601,producerpage/copy1257,Vpublish615,MMAissue303;waitQK48/PV9 | Same H1/H2, actual N128 per-phase clocks and API residency in same batch | `7eaa262a30ec483e`,`5c99583676994c2a` |

Native phase receipt97d6fa2bf2ec499b passes571/571TASK checks (560timed/fixture +11profile).
Role clocks overlap; summing producer and consumer phases would double-count time.
A:A95%t7 intervals by d1/d16/d32/v1/v16/v32 are [.994217,1.011484]/
[.994474,1.002567]/[.996945,1.000967]/[.990992,1.012954]/
[.995514,1.002352]/[.995715,1.001420]. All8seeds retained; no G0 claim from stage-only data.

| Recipe | Ratio vs control | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Whole64-key balanced d1 partial split F |d1F1.004925,95%CI[.996716,1.013134] | Less padded tile work; same336CTAs/one launch/all keys/max4tile tail; fixed prologue+merge remains | Close balancing alone as a winner; eliminate duplicated normalization and short-split merge work next | `97d6fa2bf2ec499b` |
| Quad-row/vector page lookup G |d16F1.163271[1.148947,1.177596],d32F1.182698[1.170122,1.195274] | Same bytes/splits/math, changed ownership and page vector reads; issue remains large | Tested mapping loses; original ownership with asynchronous coordinate lookahead remains | `97d6fa2bf2ec499b` |
| Native one-launch partial-input v1 P |v1F1.400567[1.391031,1.410102] | Removes FA3 prep/metadata/combine/norm boundaries, retains every key | Tested native replacement loses40%; launch count alone does not explain roof gap, lower native prologue/inner-loop cost needed | `97d6fa2bf2ec499b` |
| One-head N128 topology H1/H2 | H1v1/v16/v32F1.143737/1.241220/1.350739;H2 1.049733[1.036955,1.062511]/1.214812[1.203296,1.226327]/1.302626[1.296133,1.309120] | API confirms A1,H1/H2 2CTA/SM; half ring/256threads/producer32consumer216regs. Twice residency does not reduce total softmax/issue work | This topology loses despite achieved occupancy. Exact N128 softmax remains largest consumer phase; tensor denominator has not passed TASK yet | `97d6fa2bf2ec499b` |


### P5b coordinate lookahead and tensor denominator verdict

4df30d04a39e433e passes491TASK (480fixture/scored+11native phase gates),
8freshseeds2671223+7919*i,8cold rounds,172.8s private. Ratios new/S15 time;
95%t7 paired-seed intervals. A:A d1/d16/d32/v1/v16/v32:
.998592[.989053,1.008131]/1.000996[.998712,1.003280]/
.999269[.996782,1.001756]/.997665[.989537,1.005794]/
.996655[.994010,.999300]/.998084[.995583,1.000586]. Verify16 directional
control drift .33% is retained, not a per-load veto. Stage-only data does not
establish whole-layer G0. Exact handoff: dev/full_attention/page_ring/P4_HANDOFF.json.
S18-readable R8/D ports are5bf55205/2948c3e5 onto e04635ec; existing
PDL/K/V waits preserved, Dport native compile succeeds (32Bmain spill).
S18 composition speed/correctness remains P4, not inferred from S15.

| Recipe | Ratio vs control | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| R4 asynchronous4tile page-coordinate ring, original row ownership | d16F.936785[.933362,.940208];d32F.926585[.922969,.930200] | Shared coordinates eliminate late global-page dependency; actual3CTA/SM/splits unchanged, all keys; 1024Bshared | Both point winners to P4; R8 slightly better. Remaining KV issue/math overlap and fixed prologues remain | `4df30d04a39e433e` |
| R8 asynchronous8tile page-coordinate ring | d16F.933568(SE.001257)[.930596,.936540];d32F.923836(.001884)[.919381,.928291] | Pageissue1703->825/1697->846cycles per64tile; copywait27->176/25->142 exposes transport. Same3CTA/SM,24/2 and12/2 split/cluster; 2048Bshared | Preferred d16/d32full to P4: source17bfdfd7/patch5c1d5de7 ontoS15, CUDA hunks preserve S18 PDL. Logical3.44/3.68TB/s still below4.58arraystream; G0 composition open | `4df30d04a39e433e` |
| Dnative FP8 P*ones denominator co-issued in PV | v16F.978271(SE.001883)[.973818,.982723];v32F.982280(.001375)[.979030,.985531] | Scalar sum removed; exact N128softmax1635->1515/1602->1491cyc, MMAissue305->452/305->446. Same1CTA/SM despite known16Bmain spills; every TASK gate passes | Both points to P4 source5c8ba960/patchb4fa05ff; old extra1e-6 native oracle abort was unmeasured, now positive TASK verdict. Remaining max/exp/packing and denominator-in-original-PV remain; G0 with Vwait elision open | `4df30d04a39e433e` |
| Repeated quad-row G control |d16F1.156528[1.137802,1.175255];d32F1.188854[1.174186,1.203522] | Reproduces97d6fa2b loss while R changes only page readiness, not ownership | Close vector/ownership form, not asynchronous coordinate prefetch | `4df30d04a39e433e` |


| Recipe | Ratio vs control | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Immediate P packing into registers I | **Runtime unmeasured**; local normal+forced25fixtures eager/replay and25static pass each | Claimed live-range reduction not obtained: O3ST4 still168regs,+4Bspill,8105->8102static instructions; MUFU.EX2/F2FP/max/shuffle/add/FMA counts identical | No standalone run or P4 nomination; other layouts reducing actual softmax instructions remain | local `dev/full_attention/pack/instructions.json`,`native_I.json` |


### P5b S18+R8+D oracle and reduction verdict

Eight fresh seeds3671223+7919*i,8cold rounds,564TASK pass. Ratios new/control
time,95%t7 intervals; native output poisoning and graph-node checks pass.
A:A d1/d16/d32/v1/v16/v32 .994664[.982516,1.006812]/
1.001785[.999147,1.004424]/1.000913[.998474,1.003353]/
.998227[.982561,1.013894]/.996666[.991736,1.001596]/
1.000285[.997469,1.003101]. All contain1. Stage control is exact S18e04635ec
with R8/D; raw times17.8385/95.5865/178.6775/29.3150/131.5070/220.9800us.

| Recipe | Ratio vs control | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Initial fixed/reference+lastCTA+compact batch | **Unmeasured**; custom times .001us floor and negative prepass times | Callers cached the default stream before capture; custom kernels ran during warmup but not replay. Unpoisoned same-input checks did not detect it. Native API occupancy C3=4 remains valid separately | No point nominations. Correction selects stream inside each call, checks native kernel nodes and poisons outputs before TASK replay; numerical checks unchanged | `b9488225d13b4407` |
| Fixed-reference oracle, free exact max | v16F.934211(SE.003015)[.927081,.941341];v32F.949841(.002017)[.945070,.954611] | Remove running max/rescale, keep FP8 P/D denominator/exact final normalization. Max supplied by native all-key pass outside timing, explicitly an oracle | Never a candidate. Max/rescale removal yields only5–7% of this stage; exponent/packing and other issue remain, not all softmax algorithms bounded | `d53966e4afa949fc` |
| Paid fixed reference via native K-only pass | v16F1.424323[1.414016,1.434630];v32F1.515303[1.509933,1.520672] | Every current call recomputes true per-row max over all cached/causal new keys; separate pass71.868/133.309us. Four captured kernels versus2control | Close this two-pass recipe; no free-data reuse. Cheaper exact bounds or different softmax layouts remain | `d53966e4afa949fc` |
| Full-context nonspinning lastCTA, NK64/128 x splits16..256 | Best d1F NK128/ns64 2.061395[2.035137,2.087653];best v1F NK128/ns32 4.294637[4.253015,4.336260];all20points lose | One launch includes full history and causal8queries, native FP8, final normalization/gate. Only2final CTAs merge all rows; v1 time125.9us atns32 rises278.1/515.2 atns128/256 despite less tile work | Source+scaling implicate serialized merge (inference, not phase clocks). Close this layout; parallel DSM/native-VQ8 reduction remains; FA3five-kernel fixed work still wins | `d53966e4afa949fc` |
| C3 q1 dead-stage new-token reuse,3stages | d16F.988939(SE.002827)[.982253,.995624];d32F.990809(.004215)[.980842,1.000777] | Free4KiB dedicated new-token shared; read exact current FP8 after history drains. API R8/C3 CTAs3/4,cluster2 capacity198/264;128regs/no spills | Both points to P4, concrete source6e46ccbc/patchb2103b37 onto S18+R8/D controlbf1e3983. Four-CTA residency alone is not a large remaining lever; P4 confirms G0, no standalone small-point send | `d53966e4afa949fc` |
| C4 storage-reuse control,original4stages | d16F.993815[.990556,.997073];d32F.996489[.993764,.999213] | Same storage change, still3CTA/SM; isolates pipeline/occupancy from the new-token-buffer change | Both point winners also nominated. Remaining d32 transport approaches old same-kernel gather-only3.74TB/s versus R8~3.68; not proof of a universal random-page roof | `d53966e4afa949fc` |

Exact handoff/seed/resource/capture evidence: dev/full_attention/fixed/.
Native clocks from4df30d04 and timeline7eaa262a still attribute d1fixed
prologue/reduction and v1FA3prep/metadata/combine. The current oracle limits
max/rescale's contribution in this layout; exp/packing remain the large
verify issue term. Added occupancy gives only~1% at decode16/32.


### P5b S19 fixed phases and remaining issue

S19 shipped8b415553,expanded62ea5bf4, R8/D already applied;8fresh seeds
4671259+7919*i,8cold rounds,498native TASK pass (78official,140stress,
280scored), host TASK passes. Ratios are new/control time with95%t7 CI;
all SE/seeds/CUPTI/samples in dev/full_attention/phases/result.json.
A:A d1/d16/d32/v1/v16/v32 means1.003063/1.001887/1.001040/1.001728/
.996347/.998640, SE.004849/.002076/.001058/.004183/.001241/.001350.
v16 CI[.993413,.999281] shows-.365% mean drift; report it, not a per-load
veto (G0 uses A:A noise). No failed TASK/device/empty-capture controls.
Control means17.608/96.395/179.186/29.6155/133.198/223.689us.

Source correction: **d1 is native partial-input attention**, not FA3;
`_PP_ROWS=((1,1,False),)`, `_PP_FAST=False`. v1 FA3 launches prep, metadata,
main, combine, norm/gate. Default metadata reads current lengths and cannot
be cached as-is. A legal constant schedule can replace that policy: fixed
S, one packed M tile, batch0, heads_l2=1. Only those constants enter prepare;
FA3 computes every split's key range from fresh device lengths and combine
resets its semaphore. Prep/main/combine/norm depend on current values.
Combine is required atS>1; public fwd still executes it when returning its
partials, so reading them afterward does not remove a launch. Fusion into
projection/FA3 combine remains a kernel change, not an untimed-data cache.

| Recipe | Ratio vs control | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| P64c8 | 1x1f 1.570602[1.552734,1.588471] | Same S19 native partial kernel; split/cluster geometry only. Incumbent336CTAs;384/c16 needs24clusters versus21resident slots. | No point winner; close tested split geometry, d1 prologue/reduction remains. | `deea291e294b4a53` |
| P128c8 | 1x1f 1.120041[1.108761,1.131320] | Same S19 native partial kernel; split/cluster geometry only. Incumbent336CTAs;384/c16 needs24clusters versus21resident slots. | No point winner; close tested split geometry, d1 prologue/reduction remains. | `deea291e294b4a53` |
| P128c16 | 1x1f 1.186645[1.178075,1.195215] | Same S19 native partial kernel; split/cluster geometry only. Incumbent336CTAs;384/c16 needs24clusters versus21resident slots. | No point winner; close tested split geometry, d1 prologue/reduction remains. | `deea291e294b4a53` |
| P256c8 | 1x1f 1.000538[0.991084,1.009991] | Same S19 native partial kernel; split/cluster geometry only. Incumbent336CTAs;384/c16 needs24clusters versus21resident slots. | No point winner; close tested split geometry, d1 prologue/reduction remains. | `deea291e294b4a53` |
| P256c16 | 1x1f 1.020657[1.012105,1.029209] | Same S19 native partial kernel; split/cluster geometry only. Incumbent336CTAs;384/c16 needs24clusters versus21resident slots. | No point winner; close tested split geometry, d1 prologue/reduction remains. | `deea291e294b4a53` |
| P384c16 | 1x1f 1.547330[1.532862,1.561797] | Same S19 native partial kernel; split/cluster geometry only. Incumbent336CTAs;384/c16 needs24clusters versus21resident slots. | No point winner; close tested split geometry, d1 prologue/reduction remains. | `deea291e294b4a53` |
| M1 | 1x8f 24.135894[23.904792,24.366996] | Shape-only17int32 schedule, fresh runtime key bounds; M1 removes combine but main712.5us; other explicit counts preserve required combine. | Close tested count only; retain M64 point winner. | `deea291e294b4a53` |
| M8 | 1x8f 3.107040[3.070437,3.143643] | Shape-only17int32 schedule, fresh runtime key bounds; Static split count changes parallelism and combine work. | Close tested count only; retain M64 point winner. | `deea291e294b4a53` |
| M16 | 1x8f 1.790416[1.772784,1.808048] | Shape-only17int32 schedule, fresh runtime key bounds; Static split count changes parallelism and combine work. | Close tested count only; retain M64 point winner. | `deea291e294b4a53` |
| M32 | 1x8f 1.151979[1.138316,1.165641] | Shape-only17int32 schedule, fresh runtime key bounds; Static split count changes parallelism and combine work. | Close tested count only; retain M64 point winner. | `deea291e294b4a53` |
| M64 | 1x8f 0.909787[0.898514,0.921060] | Shape-only17int32 schedule, fresh runtime key bounds; metadata removed, four kernels. Main16.896us and exposed combine6.400us in one profile. | M64 to P4, no new CUDA units; G0 pending. Same-cap U64 comparison is not pure metadata isolation: default actual split count can differ. | `deea291e294b4a53` |
| M128 | 1x8f 1.121963[1.109381,1.134546] | Shape-only17int32 schedule, fresh runtime key bounds; Static split count changes parallelism and combine work. | Close tested count only; retain M64 point winner. | `deea291e294b4a53` |
| U64 | 1x8f 0.999652[0.990027,1.009278] | Shape-only17int32 schedule, fresh runtime key bounds; U64 keeps dynamic metadata under requested cap64. | U64 point neutral and dominated by M64; nominal point included in handoff. | `deea291e294b4a53` |
| C3 | 16x1f 0.995921[0.991601,1.000241];32x1f 0.991918[0.983318,1.000519] | Four-CTA dead-stage storage reuse, zero spills; new S19 confirmation. | Both points to P4; C3 preferred at both; G0 pending. | `deea291e294b4a53` |
| T8 | 16x1f 1.010738[1.006100,1.015375];32x1f 1.003610[0.995320,1.011901] | Vector16B page-ID copies with scalar alignment/tail path; native4CTA/SM,128regs/no spills. | No point winner; coordinate vectorization alone does not remove the remaining transport cost. | `deea291e294b4a53` |
| T16 | 16x1f 0.996973[0.991219,1.002726];32x1f 0.996044[0.992139,0.999949] | Vector pages andPD16; static shared drops native occupancy to3CTA despite128regs/no spills. | Both points to P4; C3 preferred at both; G0 pending. | `deea291e294b4a53` |
| E25 | 16x8f 1.075556[1.070841,1.080272];32x8f 1.065355[1.060803,1.069906] | On D: pure-register25% Cody-Waite polynomial/MUFU split. ST4 static MUFU492->372, total instructions7881->8713;280F2FP/190SHFL unchanged. | Closes this SFU/FMA rebalance on D; fragment/packing issue and exposed combine tail remain. v16 A:A drift-.365% disclosed. | `deea291e294b4a53` |
| E50 | 16x8f 1.221827[1.215289,1.228365];32x8f 1.236092[1.232464,1.239719] | On D: pure-register50% Cody-Waite polynomial/MUFU split. ST4 static MUFU492->252, total instructions7881->9705;280F2FP/190SHFL unchanged. | Closes this SFU/FMA rebalance on D; fragment/packing issue and exposed combine tail remain. v16 A:A drift-.365% disclosed. | `deea291e294b4a53` |

Remaining native terms: d1 single336CTA launch19.008us in one profile;
prior prologue/epilogue8751/9230 versus8179loop cycles (4df30d04).
C3 d16/d32 each512CTAs,4CTA/SM: transport/math remains; T16's small points
lose to C3. Old same-kernel gather-only3.62/3.74TB/s is not a universal roof.
v1 M64 main16.896us, combine6.400us critical increment, norm1.824us
including gap, prep1.344us. v16/v32 D main128.960/221.313us; combine
critical tails6.688/5.280us versus26.048/22.784us residence. Earlier D
softmax1515/1491 versus MMAissue452/446cycles/tile still attributes the
large verify term; polynomial split adds issue and loses. These first-seed
CUPTI spans are diagnostic, not paired timing estimates.

Local exact payload normal/forced each498TASK+16host+6static. Preferred
S19+C3/M64 layer-call patch and standalone21-line M64 patch are in
phases/P4_HANDOFF.json with hashes and component admission. P4 confirms
held-out six-load G0 and nearby performance; no stage-only promotion or
standalone small-point follow-up.

### P5b S19 all-FA3 schedules and parallel merge

Native9dc1159a076b4e6b:928TASK,8newseeds5671243+7919*i,8cold rounds;154.73s.
Every arm CI/SE, raw seed samples, CUPTI and launch counts in dev/full_attention/chain/
POINTS.md and result.json. Ratios new/S19 time. A:A v1S/v16S/v1F/v16F/v32F
.982109/1.000372/.997719/.997584/.998813,SE.005097/.003783/.001525/.001022/.000964.
v1S directional-1.79% drift retained; v16F upper CI rounds1.000000. No per-load veto.
Actual FA3 call controls cover1. S19 v1S is native sliding_main<8,4,128>,one launch,
not FA3; R2_SELECTION[(1,8)]=('dsm',4,16). Preserve P5 native/query/RAW2 paths.
Only v16S/v1F actually go FA3 five->four via shape schedules or five->three via
parallel combine+norm. Full native D two->two for row merge; DSM F two->one loses.
No FA3 main/epilogue fusion claim: main+parallel merged norm remain separate kernels.
D/M equal-split ABI controls v16S.996299[.980553,1.012045],v1F.998698[.987812,1.009585].
S/M aliases and identity-S points are controls, not new shader nominations.

| Recipe | Ratio vs control | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| First prepared-schedule native attempt | Unmeasured, pre-timing TASK path stopped | At short cache capacity FA3 normalizes local to full/causal and needs4 metadata vectors, not3. Fixed from shapes; actual setter/layout probed locally | No recipe closed; all gates retained | `b8ddced394bd46ae` |
| Second parallel-merge native attempt | Unmeasured, correctness stopped timing | Empty-tile G control subtracted cta_of(0)>0, writing negative slots; G8/B32 also exceeded132CTA workspace and corrupted later arms. Fixed slot=s*CS+rank, max(SMs,8*segments)+segments storage and undefined empty FA3 partial masking | No loss/promotion claim. CPU2054partition cases and all-rank G local proof now pass; final928nativeTASK passes | `9c577135f9d64269` |
| M1 | v1S 2.677961; v16S 1.388207 | Literal-shape fixed split schedule, fresh key bounds, no metadata launch; required combine/reset retained | Nominate correct points<1; v1S diagnostics all lose to native. Other counts only close themselves | `9dc1159a076b4e6b` |
| M2 | v1S 2.070571; v16S 1.134263 | Literal-shape fixed split schedule, fresh key bounds, no metadata launch; required combine/reset retained | Nominate correct points<1; v1S diagnostics all lose to native. Other counts only close themselves | `9dc1159a076b4e6b` |
| M4 | v1S 1.696964; v16S 0.953395[0.940476,0.966313] | Literal-shape fixed split schedule, fresh key bounds, no metadata launch; required combine/reset retained | Nominate correct points<1; v1S diagnostics all lose to native. Other counts only close themselves | `9dc1159a076b4e6b` |
| M8 | v1S 1.475529; v16S 1.083403 | Literal-shape fixed split schedule, fresh key bounds, no metadata launch; required combine/reset retained | Nominate correct points<1; v1S diagnostics all lose to native. Other counts only close themselves | `9dc1159a076b4e6b` |
| M16 | v1S 1.398761 | Literal-shape fixed split schedule, fresh key bounds, no metadata launch; required combine/reset retained | Nominate correct points<1; v1S diagnostics all lose to native. Other counts only close themselves | `9dc1159a076b4e6b` |
| R1 | v1S 2.626274; v16S 1.416199 | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R2 | v1S 1.942622; v16S 1.105790 | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R4 | v1S 1.547741; v16S 0.908322[0.897655,0.918988] | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R8 | v1S 1.388184; v16S 1.027033 | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R16 | v1S 1.359413 | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R4w4 | v1S 1.551220; v16S 0.890066[0.878445,0.901687] | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| M32 | v1F 1.149490 | Literal-shape fixed split schedule, fresh key bounds, no metadata launch; required combine/reset retained | Nominate correct points<1; v1S diagnostics all lose to native. Other counts only close themselves | `9dc1159a076b4e6b` |
| M64 | v1F 0.904360[0.893519,0.915200] | Literal-shape fixed split schedule, fresh key bounds, no metadata launch; required combine/reset retained | Nominate correct points<1; v1S diagnostics all lose to native. Other counts only close themselves | `9dc1159a076b4e6b` |
| M128 | v1F 1.116154 | Literal-shape fixed split schedule, fresh key bounds, no metadata launch; required combine/reset retained | Nominate correct points<1; v1S diagnostics all lose to native. Other counts only close themselves | `9dc1159a076b4e6b` |
| R32 | v1F 1.100985 | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R64 | v1F 0.888090[0.875082,0.901098] | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R128 | v1F 1.096249 | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R64w4 | v1F 0.890120[0.879146,0.901094] | Installed native FA3 main + one/four-warp row combine/norm, three launches; all keys | Nominate points<1; best v16 R4w4 and v1 R64. True fused FA3 main/epilogue remains open | `9dc1159a076b4e6b` |
| R | v16F 0.992308[0.984905,0.999711]; v32F 1.005864 | Original D main, parallel row max/exp reused across64dims, norm/gate fused; one/four-warps per CTA, two launches | Both v16 points and v32 Rw4 point to P4; main issue remains, no stage-only G0 claim | `9dc1159a076b4e6b` |
| Rw4 | v16F 0.992111[0.986609,0.997613]; v32F 0.997763[0.994638,1.000888] | Original D main, parallel row max/exp reused across64dims, norm/gate fused; one/four-warps per CTA, two launches | Both v16 points and v32 Rw4 point to P4; main issue remains, no stage-only G0 claim | `9dc1159a076b4e6b` |
| G4 | v16F 1.573412; v32F 1.721246 | Fixed4/8CTA clusters per segment, original combine, same native D math | Close fixed per-segment geometry: imbalance/underfill/waves versus globally balanced S19; not DSM generally | `9dc1159a076b4e6b` |
| F4 | v16F 1.581045; v32F 1.763011 | Same G partition, all384threads in every CTA reduce through DSM; dead-stage shared reuse, one launch | F/G v16 1.004846/1.033585,v32 1.024264/1.029316: all CIs>1. Close these fixed-cluster fused forms only | `9dc1159a076b4e6b` |
| G8 | v16F 1.540363; v32F 1.371561 | Fixed4/8CTA clusters per segment, original combine, same native D math | Close fixed per-segment geometry: imbalance/underfill/waves versus globally balanced S19; not DSM generally | `9dc1159a076b4e6b` |
| F8 | v16F 1.592101; v32F 1.411737 | Same G partition, all384threads in every CTA reduce through DSM; dead-stage shared reuse, one launch | F/G v16 1.004846/1.033585,v32 1.024264/1.029316: all CIs>1. Close these fixed-cluster fused forms only | `9dc1159a076b4e6b` |

Preferred P4 combined S19 patch5f18b0a5/source dbeeb847: v16S R4w4
.890066[.878445,.901687],SE.004914; v1F R64.888090[.875082,.901098],SE.005501;
v16F Rw4.992111[.986609,.997613],SE.002327;v32F.997763[.994638,1.000888],SE.001322.
R4w4/M4.933739[.918506,.948972]; R64/M64.982057[.970121,.993992].
Standalone FA3-only ef9e9be4, native-row-only41e683b8, metadata-only ce4a7609;
all apply independently onto S19 8b415553, not concatenated prepare wrappers.
No P5 code changes except its authorized README record. Full G0/prepare cost stays P4.
Native F main168regs/24Bspill versus G168/32B; FA reducer43 and VQ56regs/no spills.
Single-profile critical increments v1F M64 1.66+17.70+6.66+1.70 -> R64
1.31+17.63+8.10us; v16F A129.18+6.46 -> Rw4 129.47+5.63, v32 A221.05+5.31 ->
Rw4 220.32+4.99us. Main issue remains largest. Dependent27us duration includes
PDL wait, not27us exposed merge. G4/G8 main alone v16 207.74/202.43 andv32
384.35/303.55 versus A129.18/221.05us separates geometry loss from extra DSM cost.
Normal+forced exact payload each928TASK+16host/6static, actual transport7checks;
selected candidate admission and exact patches in dev/full_attention/chain/P4_HANDOFF.json.

### P5b licensed public attention ports on S19

R-20261026-1, read-only external pinned clone review before use; no donor
scripts/install/tests/notebooks executed. Operatorstack6d9b7765 MIT and
Lee601c6541 v56 Apache notices embedded in the exact independent S19
patch. Gluon/warp-specialization toolchain present. All12cells, **704native
TASK gates**,8freshseeds x8symmetric cold rounds, all3072roundns kept.
Per-seed ratios from all rounds;95%t7, no filtering. Every A:A CI covers1,
point range .989612–1.009940 (d1 has±3.9% uncertainty). Component nomination
only; P4 held-out whole-layer G0/prepare<=300s remains unmeasured.

| Recipe | Port/S19 ratio [95% CI] | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Licensed public 1x1s (O/Lee) | O 1.262517[1.222107,1.302928]; L 1.232249[1.194636,1.269863] | Both-head block-diagonal decode; OP keys-on-M for full; Lee native fdec/attnd, fused norm/RoPE/KV/merge | Retain S19; closes only these pinned dispatches, not all transport/softmax recipes | `26d5054ffce94231` |
| Licensed public 1x1f (O/Lee) | O 1.028181[1.016495,1.039868]; L 1.008982[0.999721,1.018243] | Both-head block-diagonal decode; OP keys-on-M for full; Lee native fdec/attnd, fused norm/RoPE/KV/merge | Retain S19; closes only these pinned dispatches, not all transport/softmax recipes | `26d5054ffce94231` |
| Licensed public 16x1s (O/Lee) | O 0.996939[0.963572,1.030305]; L 0.954256[0.919953,0.988558] | Both-head block-diagonal decode; OP keys-on-M for full; Lee native fdec/attnd, fused norm/RoPE/KV/merge | Preferred L; per-cell point nominations, actual whole-layer/PDL cost unmeasured | `26d5054ffce94231` |
| Licensed public 16x1f (O/Lee) | O 0.991556[0.985235,0.997878]; L 1.112085[1.103138,1.121032] | Both-head block-diagonal decode; OP keys-on-M for full; Lee native fdec/attnd, fused norm/RoPE/KV/merge | Preferred O; per-cell point nominations, actual whole-layer/PDL cost unmeasured | `26d5054ffce94231` |
| Licensed public 32x1s (O/Lee) | O 0.956675[0.940380,0.972970]; L 0.860893[0.845972,0.875814] | Both-head block-diagonal decode; OP keys-on-M for full; Lee native fdec/attnd, fused norm/RoPE/KV/merge | Preferred L; per-cell point nominations, actual whole-layer/PDL cost unmeasured | `26d5054ffce94231` |
| Licensed public 32x1f (O/Lee) | O 1.026920[1.022488,1.031352]; L 1.107322[1.100255,1.114388] | Both-head block-diagonal decode; OP keys-on-M for full; Lee native fdec/attnd, fused norm/RoPE/KV/merge | Retain S19; closes only these pinned dispatches, not all transport/softmax recipes | `26d5054ffce94231` |
| Licensed public 1x8s (O/Lee) | O 0.780899[0.773075,0.788723]; L 1.105711[1.089843,1.121578] | OP last-arriver Triton or Gluon producer/consumer; Lee page-row producer +2QK/PV consumer WGs, tensor-core P sums, balanced splits/early query loads | Preferred O; per-cell point nominations, actual whole-layer/PDL cost unmeasured | `26d5054ffce94231` |
| Licensed public 1x8f (O/Lee) | O 1.015518[1.009180,1.021855]; L 0.757719[0.752012,0.763426] | OP last-arriver Triton or Gluon producer/consumer; Lee page-row producer +2QK/PV consumer WGs, tensor-core P sums, balanced splits/early query loads | Preferred L; per-cell point nominations, actual whole-layer/PDL cost unmeasured | `26d5054ffce94231` |
| Licensed public 16x8s (O/Lee) | O 0.779475[0.766451,0.792498]; L 0.763592[0.752210,0.774975] | OP last-arriver Triton or Gluon producer/consumer; Lee page-row producer +2QK/PV consumer WGs, tensor-core P sums, balanced splits/early query loads | Preferred L; per-cell point nominations, actual whole-layer/PDL cost unmeasured | `26d5054ffce94231` |
| Licensed public 16x8f (O/Lee) | O 1.138588[1.128975,1.148201]; L 0.968115[0.961687,0.974543] | OP last-arriver Triton or Gluon producer/consumer; Lee page-row producer +2QK/PV consumer WGs, tensor-core P sums, balanced splits/early query loads | Preferred L; per-cell point nominations, actual whole-layer/PDL cost unmeasured | `26d5054ffce94231` |
| Licensed public 32x8s (O/Lee) | O 0.799316[0.788602,0.810030]; L 0.750479[0.742199,0.758760] | OP last-arriver Triton or Gluon producer/consumer; Lee page-row producer +2QK/PV consumer WGs, tensor-core P sums, balanced splits/early query loads | Preferred L; per-cell point nominations, actual whole-layer/PDL cost unmeasured | `26d5054ffce94231` |
| Licensed public 32x8f (O/Lee) | O 1.377356[1.367078,1.387634]; L 1.076995[1.073106,1.080885] | OP last-arriver Triton or Gluon producer/consumer; Lee page-row producer +2QK/PV consumer WGs, tensor-core P sums, balanced splits/early query loads | Retain S19; closes only these pinned dispatches, not all transport/softmax recipes | `26d5054ffce94231` |

The S19 ABI is FP32 split planes1/10/7/3 or BF16, width1280(QKV-only)
or2304(QKV+gate). Sum/cast/pad is timed, +inf internal gate preserves S19's
separate real gate. Native attention/merge1launch; FP32pack adds1. S19→port
launches v1F/v16S5→2; v1S1→2 yet OP wins22%; v32S1→1, Lee wins25%.
Full32 neither wins despite one fewer launch. No work/values moved to prepare.

All12correct point<1 pieces(6O/6L), SE/seeds/A:A and8preferred cells in
`public_ports/P4_HANDOFF.json`. Combined patch**97c86143**, source**4a60c4e3**,
applies onto S19**8b415553**; alternative OP/Lee patches84c9e409/299d9d8f.
Local normal/forced704TASK+16host each, final wrapper28eager+graph cases,
reviewed kernels byte/AST provenance and sm90a compilation pass. Operator
full verify emits13WGMMA; full decode `mma_v2` is not a native-WGMMA claim.
Selected cells bypass older FA3/row-merge interceptors; P4 compares d16F OP
point to its C3 variant. P5 producer_finish README records OP v1S fusedprep
winner so the same fusion is not built twice. No further run queued.
Lab-only host intentionally ends at prepare after complete evidence; outer
`passed:false` is not a failed704TASK gate. [Evidence](../problems/muse_attention_decode/dev/public_ports/README.md).

### P5b post-port direct-input and consumer hybrids

Post-port picks26d5054f,8freshseeds10739123+7919*i, all4752 cold integer
round times retained,2N cyclic/reverse rounds,780nativeTASK pass.95%t7,
no filtering. All A:A CIs cover1 except v1F .993350[.987504,.999196];
disclose drift, no precise marginal closure there. No whole-layer G0 claim.

| Recipe | New/pick [95% CI] | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| Post-port 1x1s F | 1.068873[1.056879,1.080867] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 1x1f F | 1.292213[1.273897,1.310530] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 16x1s O | 1.294585[1.276435,1.312734] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 16x1f K | 1.341944[1.336264,1.347624] | Lee K-major native FP8 QK/f16 PV on full decode | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 32x1s F | 0.985855[0.960184,1.011527] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | P4 nominee, held-out whole-layer/prepare confirmation | `cde88b9136de4093` |
| Post-port 32x1f F | 1.130155[1.125569,1.134742] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 1x8s F | 0.839438[0.819888,0.858987] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | P4 nominee, held-out whole-layer/prepare confirmation | `cde88b9136de4093` |
| Post-port 1x8f F | 1.402405[1.388335,1.416474] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 16x8s O | 1.024066[1.016012,1.032119] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 16x8f F | 1.379079[1.374989,1.383169] | Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 32x8s E | 1.126020[1.108130,1.143910] | Lee direct-input +25% polynomial exponents | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |
| Post-port 32x8f Q | 1.066325[1.060408,1.072243] | Lee direct-input + early Q | Retain pick; closes this exact reader/geometry, not the mechanism family | `cde88b9136de4093` |

Largest modeled remaining cells to1.3KV: d1S347.83us/load/R1.0121;
v1S432.79/1.0095; d16S506.92/1.0073; d1F198.00/1.0068;
v16S745.19/1.0064; v1F272.50/1.0059; v16F667.48/1.0057.
These combine post-port component times with S19 CUPTI9c16c850 non-attention
wall and actual old-seed KV draws; ranking counterfactual, not measured R.
Full d32F is last41.45us/load/R1.0005 despite180us kernel time.

Generic scalar QView loses2.805/3.059 at d16S/v16S: repeated FP32 queries
10.49/25.17MB versus oldBF16 .52/4.19MB, scalar per-component/runtime-plane
loops and some verify spills; exclusive issue/occupancy cost unmeasured.
Vectorized plane-outer reader remains open. Producer-V, earlyQ, poly25 and
combined lose against picks; some relative-F points improve but are not
measured original-port nominees. Full K-major loses1.342/1.196 d16F/d32F.

Combined exact licensed patch**9b043487**, source**fbaf2793**, onto current
P4S21**eacc087b**/expanded109c0135; v1S-only80e45c72, d32S-onlya59cb4b2.
Current P4 manifest/source newer than its PLAN row; C3d16F/Rw4v32F comparator
differences do not affect changed cells. Only selected families ship, same
OP kernel AST and LeeSASS except proved equivalent shared-base relocation,
Lee80regs/no spills. Final normal/forced28eager+graph cases each pass.
P4 confirms actual full prepare and held-out G0. First1c5f639111fa4f80
unmeasured due13150-byte result; exact timestamp codec repair retains all
samples/no gate changed. [Evidence](../problems/muse_attention_decode/dev/public_hybrid/README.md).

### P5b production graph transfer and vectorized direct partials

S21 native receipt changes the inference from isolated attention-call timings:
use production `kernel_fn` graphs, with prep, gate and projections inside the
boundary. Exact source audit and seed/CI reconstruction: [public_vector](../problems/muse_attention_decode/dev/public_vector/README.md).
Ratios here are **port/S19 whole-load time**, smaller is faster. The old grouped
no_PUBLIC arm isolates one port only at d16, d32 and v32; two-cell v1/v16 effects
cannot be separated from that receipt.

| Recipe | Ratio versus control | Mechanism / evidence | Leaves open | Receipt |
|---|---|---|---|---|
| d16S public port in production S21 | 1.043149 [1.042282,1.044018],64seeds; A:A .999769 [.999095,1.000444] | Ordinary pack/main remove S19 entry-release edge to independent gate-split GEMM; source proves the serialization edge, not its microseconds | D(PDL) versus V(vector direct) and combined F isolate the timing cause; exact S19 restore711b3075 already P4's | 5864563117fd44b6 |
| d32S port before9b043487 | .994855 [.993713,.995997],64seeds; A:A1.000041 [.999549,1.000533] | Small actual load gain, much less than isolated .861 call point; port main releases at exit | Accepted direct-input cell changes this baseline, so remeasure onfbaf2793 | 5864563117fd44b6 |
| v32S port in production S21 | .996492 [.994566,.998422],64seeds; A:A .999377 [.998992,.999762] | BF16 view has no pack; ordinary public main, upstream Lt/downstream O boundaries limit PDL leverage | Vectorized once-per-CTA Q/own-KV and actual native neighbour trace; disclose directional A:A drift | 5864563117fd44b6 |
| grouped public leave-out | no_PUBLIC/S21 .998709 [.997343,1.000077]; v1 1.023489, v16 .995999 | Per-call gains do not establish a net in-layer port gain; v1S/F and v16S/F are not isolated | Seven independent S19 cell restores in new batch; no per-call nomination claim | 5864563117fd44b6 |

D adds PDL-enabled launches, entry release and a wait after independent metadata/
CTA setup; Lee history producer proceeds while consumers wait. V uses 16-byte
plane loads and one query/own-KV global reader per CTA; F composes both. Native
d1S edits the actual last-CTA `_s16_global_C128_cs_CUDA`, and d1F keeps its native
launch/dispatch. Both normal/forced production screens pass108 TASK each; exact
ready candidate passes28 eager+graph each, five exact sm90a units compile and
eight native-reader cases are bit-exact. Verify spills remain; native two-lane
correctness and performance are unmeasured, not a closed recipe.

One check is queued as sender3831549, attention host, priority2, not-before
**2026-10-27 09:00KST /1793059200**: eight fresh seeds, eight symmetric cold rounds,
432TASK, all production-graph round times and adjacent-kernel profiles retained.
Payloadb4a5ec1b; controlfbaf2793=S21eacc087b+9b043487. Unsent senders3620849/3705999 were
replaced before dispatch to keep S19 caches source-owned and guard balanced
cross-request/idle lanes. Exact two-lane prologue tests pass15cases/99crossings/1idle-peer
bit-exact; full native attention TASK remains pending. All-F review
patch6a605fa2/sourcea4c63f0b is ready and unmeasured; no new fix is nominated.
P4 confirms correct whole-load improving points on disjoint held-out G0 seeds.

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
| P12 S15 attention component | SGLang/ours 1.64698 [1.62228,1.67169]; ours/leader 1.79873; A:A unavailable | Frozen S15 path behind the board ABI; official paired SGLang control | 16 tests/12 shapes pass. Leader per-shape times unpublished; aggregate gap only; no layer promotion. | 4e108beb9a0a4cdb |
| P12 S15 projection component | SGLang/ours 1.20236 [1.18432,1.22039]; ours/leader 1.36125; A:A unavailable | Frozen S15 path behind the board ABI; official paired SGLang control | 9 tests/30 shapes pass. Leader per-shape times unpublished; aggregate gap only; no layer promotion. One-reducer board adapter independently measured and admitted; in-layer fused consumers unchanged. | 08cc94050a4e4564 |
| P12 S15 moe component | SGLang/ours 2.02924 [1.99880,2.05968]; ours/leader 1.25835; A:A unavailable | Frozen S15 path behind the board ABI; official paired SGLang control | 5 tests/6 shapes pass. Leader per-shape times unpublished; aggregate gap only; no layer promotion. | 97f5e7f3c3fc4159 |
| P10 pool sparse batch | unmeasured: failed or incomplete control/gate | No recipe loss or promotion inferred | LAB_ERROR CompilationError: at 6:4:; AttributeError("'NoneType' object has no attribute 'startswith'"); AttributeError("'NoneType' object has no attribute 'startswith'"); LAB_ERROR host supplies weights=None; payloads needing weights must synthesize them (state is also None on import-time hosts) | b02c4f27e71c4340 |
| P10 pool T1 pool_router w1k8raw | 0.91739 [0.90903,0.92697]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | P4 whole-layer candidate; require paired layer gain and preserve fusions/concurrency; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w1k8sp | 2.06337 [2.05025,2.07359]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w1k8vec | 1.76780 [1.74731,1.78345]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w2k8raw | 0.96752 [0.95474,0.97758]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | unresolved within control spread; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w2k8sp | 2.34220 [2.32950,2.36173]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w2k8vec | 2.04663 [2.03082,2.06537]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w1k4raw | 1.20799 [1.18351,1.25144]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w1k4sp | 3.05926 [3.04310,3.07881]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w1k4vec | 2.35645 [2.31541,2.38773]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w1k4s4raw | 1.06273 [1.04029,1.09770]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader; S4 DSM | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w1k4s4sp | 1.68637 [1.66834,1.70427]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion; S4 DSM | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T1 pool_router w1k4s4vec | 1.52596 [1.51328,1.54337]; AA max 0.03807 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion; S4 DSM | loss; closes only this reader/structure combination, not lossless coding; random timing with separate one-hot probes | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k8raw | 0.81018 [0.79377,0.83539]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | P4 whole-layer candidate; require paired layer gain and preserve fusions/concurrency | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k8sp | 1.81691 [1.79102,1.86520]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k8vec | 1.55918 [1.53525,1.62735]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w2k8raw | 0.84406 [0.82907,0.87492]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | P4 whole-layer candidate; require paired layer gain and preserve fusions/concurrency | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w2k8sp | 2.04302 [2.01372,2.11795]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w2k8vec | 1.79145 [1.75945,1.85288]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k4raw | 1.05943 [1.03286,1.08879]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | unresolved within control spread; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k4sp | 2.67707 [2.63298,2.76928]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k4vec | 2.05746 [2.02604,2.12573]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k4s4raw | 0.99471 [0.98016,1.01815]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader; S4 DSM | unresolved within control spread; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k4s4sp | 1.47356 [1.45415,1.50940]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion; S4 DSM | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T8 pool_router w1k4s4vec | 1.34194 [1.32626,1.38950]; AA max 0.04361 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion; S4 DSM | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k8raw | 0.84632 [0.81116,0.86985]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | P4 whole-layer candidate; require paired layer gain and preserve fusions/concurrency | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k8sp | 1.80438 [1.75659,1.84901]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k8vec | 1.64560 [1.60147,1.69930]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w2k8raw | 1.22746 [1.19889,1.26552]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w2k8sp | 2.76941 [2.70229,2.83505]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w2k8vec | 2.34723 [2.29675,2.39859]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k4raw | 1.10104 [1.07541,1.12348]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k4sp | 2.37489 [2.32427,2.41459]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k4vec | 2.30047 [2.24218,2.33717]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k4s4raw | 0.94265 [0.90803,0.95941]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader; S4 DSM | unresolved within control spread; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k4s4sp | 1.54837 [1.49908,1.59214]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion; S4 DSM | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T16 pool_router w1k4s4vec | 1.42952 [1.37768,1.48196]; AA max 0.04509 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion; S4 DSM | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k8raw | 1.36835 [1.34463,1.39066]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k8sp | 2.92551 [2.89355,2.94885]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k8vec | 2.42644 [2.40000,2.45508]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w2k8raw | 1.38977 [1.37871,1.40918]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w2k8sp | 3.12944 [3.09484,3.15138]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w2k8vec | 2.62768 [2.60000,2.65596]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k4raw | 1.20529 [1.18839,1.22295]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k4sp | 2.57795 [2.55419,2.61140]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k4vec | 2.40431 [2.36387,2.43316]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k4s4raw | 1.31121 [1.29548,1.32241]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; original12-bit reader; S4 DSM | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k4s4sp | 1.91001 [1.88774,1.92787]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; scalar sparse expansion; S4 DSM | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 pool T32 pool_router w1k4s4vec | 2.20588 [2.18129,2.23067]; AA max 0.02163 | NF weights-on-M warp MMA, fused N4608 route/norm; parallel sparse expansion; S4 DSM | loss; closes only this reader/structure combination, not lossless coding | bd3f8fc778ca4832 |
| P10 all sparse batch | unmeasured: failed or incomplete control/gate | No recipe loss or promotion inferred | No LAB_BEGIN marker or comparison rows retained in receipt; no lab result can be inferred from host score. | 936dd5d5b1614711 |
| P10 all sparse batch | unmeasured: failed or incomplete control/gate | No recipe loss or promotion inferred | LAB_ERROR RuntimeError: sparse O32 WGMMA 1;     if rc:raise RuntimeError('sparse O32 WGMMA '+str(rc)); RuntimeError: sparse O32 WGMMA 1; LAB_ERROR host supplies weights=None; payloads needing weights must synthesize them (state is also None on import-time hosts) | 81592e51063b4c26 |
| P10 all-role partial batch | 69 complete cells measured; failed O32 and later cells unmeasured | Per-cell rows are emitted only after all eight seed/control gates complete | O32 host attribute and map calls succeeded; kernel launch returned30001. Removing the zero map did not fix it. No extra launch repair is justified by the established coding losses. | 50a4af4e637f4058 |
| P10 all T1 gate B | 26.60810 [26.21930,27.08120]; AA max 0.03370 | Scalar point decoder in gate GEMV; original last-split wait | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T1 o B | 2.90920 [2.83110,2.98880]; AA max 0.07580 | Sparse words reconstructed before existing NF decoder/MMA | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T1 pool_router B | 2.86810 [2.81340,2.92630]; AA max 0.02680 | Sparse words reconstructed in current CD ring | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T1 pooler B | 9.06080 [8.86930,9.25620]; AA max 0.03510 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T1 qkv_gate B | 2.50100 [2.48280,2.55260]; AA max 0.02050 | Sparse words reconstructed in current CD ring | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T1 router_hi B | 20.64220 [20.42860,21.29940]; AA max 0.03500 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T1 router_lo B | 20.43550 [19.71580,20.91490]; AA max 0.02620 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T1 router_sum B | 20.72980 [20.48910,21.00000]; AA max 0.02760 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T1 unpooler B | 9.27210 [9.16230,9.56670]; AA max 0.05990 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T8 gate B | 24.90680 [23.61240,25.44240]; AA max 0.05180 | Scalar point decoder in gate GEMV; original last-split wait | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T8 o B | 3.19180 [3.06180,3.27540]; AA max 0.04210 | Sparse words reconstructed before existing NF decoder/MMA | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T8 pool_router B | 13.07550 [12.75410,13.34080]; AA max 0.04670 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T8 pooler B | 8.75220 [8.61960,8.93150]; AA max 0.02430 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T8 qkv_gate B | 10.63530 [10.24080,10.77040]; AA max 0.03530 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T8 router_hi B | 19.94750 [19.22410,20.40950]; AA max 0.03780 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T8 router_lo B | 20.44040 [19.75140,20.78130]; AA max 0.08550 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T8 router_sum B | 19.48840 [18.88210,19.68890]; AA max 0.02880 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T8 unpooler B | 9.16220 [8.96610,9.56920]; AA max 0.05110 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T16 gate B | 25.54850 [24.65170,26.21180]; AA max 0.04490 | Scalar point decoder in gate GEMV; original last-split wait | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T16 o B | 2.76520 [2.68160,2.92520]; AA max 0.03420 | Sparse words reconstructed before existing NF decoder/MMA | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T16 pool_router B | 13.63170 [13.28120,13.86780]; AA max 0.02090 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T16 pooler B | 8.84130 [8.54170,9.12400]; AA max 0.03760 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T16 qkv3 B | 10.32000 [9.71410,10.44220]; AA max 0.05050 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T16 qkv_gate B | 10.88400 [10.61950,10.97210]; AA max 0.03540 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T16 router_hi B | 19.81080 [18.70830,20.00740]; AA max 0.07410 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T16 router_lo B | 19.75850 [19.41130,20.11250]; AA max 0.06900 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T16 router_sum B | 19.78000 [19.38260,20.14220]; AA max 0.02990 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T16 unpooler B | 2.50500 [2.46070,2.58460]; AA max 0.07040 | Sparse words reconstructed before existing NF decoder/MMA | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T32 gate B | 18.23010 [17.79030,18.65070]; AA max 0.05070 | Scalar point decoder in gate GEMV; original last-split wait | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T32 pool_router B | 10.79200 [10.38930,10.91290]; AA max 0.03240 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T32 pooler B | 10.28820 [10.13470,10.50640]; AA max 0.03450 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T32 router_hi B | 14.41820 [13.96370,14.66310]; AA max 0.03400 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T32 router_lo B | 14.47040 [14.31590,14.68740]; AA max 0.04320 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T32 router_sum B | 14.22070 [13.99040,14.64010]; AA max 0.02700 | Scalar point decoder in current balanced/bal2/GEMV loop | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 gate B | 2.50920 [2.48920,2.52220]; AA max 0.02490 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 gate C | 20.75450 [20.61150,20.94190]; AA max 0.02490 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 o B | 5.44870 [4.98970,5.59690]; AA max 0.08910 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T128 o C | 21.51480 [19.74290,21.95730]; AA max 0.08910 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T128 pool_router B | 6.38800 [6.29950,6.48250]; AA max 0.01890 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T128 pool_router C | 17.45140 [17.23170,17.69570]; AA max 0.01890 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T128 pooler B | 5.83370 [5.65500,5.86870]; AA max 0.03610 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 pooler C | 17.74020 [17.20730,17.87800]; AA max 0.03610 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 router_hi B | 2.21570 [2.16630,2.26430]; AA max 0.04670 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 router_hi C | 55.06590 [53.22720,55.49680]; AA max 0.04670 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 router_lo B | 2.21980 [2.16100,2.23720]; AA max 0.04940 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 router_lo C | 54.83800 [53.66010,56.25470]; AA max 0.04940 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 router_sum B | 2.02930 [1.99240,2.08070]; AA max 0.03770 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 router_sum C | 51.33830 [50.30280,52.41680]; AA max 0.03770 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T128 unpooler B | 28.36130 [28.16790,28.63990]; AA max 0.02000 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T128 unpooler C | 18.11960 [17.92010,18.23920]; AA max 0.02000 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T256 gate B | 2.25200 [2.22270,2.26750]; AA max 0.01100 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 gate C | 16.16900 [15.86250,16.31500]; AA max 0.01100 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 o B | 4.60120 [4.46580,4.64250]; AA max 0.05480 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T256 o C | 17.87460 [17.45210,18.16810]; AA max 0.05480 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T256 pool_router B | 5.32650 [5.23620,5.43640]; AA max 0.02990 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T256 pool_router C | 24.61250 [24.12070,25.02450]; AA max 0.02990 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T256 pooler B | 5.40540 [5.18140,5.46340]; AA max 0.05050 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 pooler C | 15.79810 [15.22980,16.02790]; AA max 0.05050 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 qkv_gate B | 4.76050 [4.67200,4.79770]; AA max 0.03140 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T256 qkv_gate C | 22.29950 [21.91720,22.52690]; AA max 0.03140 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T256 router_hi B | 2.04560 [2.01540,2.09770]; AA max 0.02970 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 router_hi C | 43.44510 [43.04100,44.08900]; AA max 0.02970 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 router_lo B | 2.03390 [1.99670,2.04400]; AA max 0.03160 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 router_lo C | 44.43720 [43.79530,44.69100]; AA max 0.03160 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 router_sum B | 1.89750 [1.88430,1.92160]; AA max 0.01700 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 router_sum C | 42.06180 [41.60490,42.48270]; AA max 0.01700 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding; standalone diagnostic | 50a4af4e637f4058 |
| P10 all T256 unpooler B | 6.31950 [6.02600,6.38870]; AA max 0.03170 | Timed restore plus current library; full scratch traffic included | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T256 unpooler C | 27.88580 [26.48250,28.22160]; AA max 0.03170 | Direct VDSK, timed sparse decode inside GEMM | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
| P10 all T1 qkv3 B | 3.00870 [2.83580,3.08430]; AA max 0.05760 | Sparse words reconstructed before existing NF decoder/MMA | loss; closes only this reader/structure combination, not lossless coding | 50a4af4e637f4058 |
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
| P10 grouped Triton point reader | unmeasured: T256 fused-route equality gate failed | Grouped codewords/masks per1024W inside original dot loops | Native ten-cell section is independently gated and valid; grouped retry 83c65f7b5b84402f supplies the fresh measured comparison after tie diagnosis | 3a91afeed9ea47fd |
| P10 grouped retry T8 qkv_gate | group 5.4249 [5.3381,5.5328]; scalar 10.5388; AA 0.0154 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T16 qkv_gate | group 5.5560 [5.4504,5.6960]; scalar 10.8036; AA 0.0534 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T16 qkv3 | group 5.5683 [5.4290,5.5935]; scalar 10.4679; AA 0.0372 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T8 pool_router | group 6.6203 [6.5892,6.6601]; scalar 12.9320; AA 0.0150 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T16 pool_router | group 6.8934 [6.7126,6.9763]; scalar 13.4537; AA 0.0278 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T32 pool_router | group 7.1236 [6.8883,7.1956]; scalar 10.6839; AA 0.0428 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T1 unpooler_slice | group 4.6459 [4.5585,4.7512]; scalar 9.3846; AA 0.0438 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T8 unpooler_slice | group 4.3193 [4.2780,4.4454]; scalar 8.8310; AA 0.0290 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T1 gate | group 11.6841 [11.4863,11.9295]; scalar 27.2426; AA 0.0428 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T8 gate | group 11.3644 [10.8843,11.4392]; scalar 25.8089; AA 0.0298 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T16 gate | group 11.0408 [10.7182,11.2487]; scalar 25.9014; AA 0.0290 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T32 gate | group 9.5894 [9.4426,9.7932]; scalar 18.5158; AA 0.0293 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T128 qkv_gate | group 7.2305 [6.9544,7.3923]; scalar 23.5701; AA 0.0655 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T128 o | group 14.5604 [14.2437,14.8971]; scalar 22.3008; AA 0.0742 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T128 pool_router | group 6.5346 [6.4323,6.5835]; scalar 17.4388; AA 0.0213 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T128 unpooler | group 7.2816 [7.1988,7.3768]; scalar 18.1502; AA 0.0392 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T256 qkv_gate | group 8.9025 [8.7490,8.9729]; scalar 22.3895; AA 0.0220 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T256 o | group 11.7326 [11.3199,11.8621]; scalar 17.9479; AA 0.0386 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T256 pool_router | group 9.4952 [9.3197,9.6434]; scalar 24.0194; AA 0.0223 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P10 grouped retry T256 unpooler | group 10.3825 [10.2606,10.4642]; scalar 27.1862; AA 0.0183 | Shared1024W codewords/masks in original dot geometry | Measured loss closes this grouped schedule. All TASK-derived FP64 route/norm gates pass; numerical near-tie membership allowed, not exact order. No P4 promotion. | 83c65f7b5b84402f |
| P12 output reducer T1 qkv_gate | B/A 0.7333 [0.7268,0.7427]; P/A 0.6352; AA 0.0136 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T1 o | B/A 0.9708 [0.9567,0.9978]; P/A 0.8094; AA 0.0432 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T1 pooler | B/A 0.7892 [0.7804,0.8038]; P/A 0.7206; AA 0.0192 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T1 unpooler | B/A 0.6601 [0.6404,0.6806]; P/A 0.5077; AA 0.0404 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T1 router | B/A 0.5739 [0.5604,0.5888]; P/A 0.4678; AA 0.0362 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T8 qkv_gate | B/A 0.7878 [0.7793,0.8089]; P/A 0.7045; AA 0.0226 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T8 o | B/A 1.0000 [0.9495,1.0160]; P/A 0.8016; AA 0.0505 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T8 pooler | B/A 0.8749 [0.8526,0.8931]; P/A 0.7993; AA 0.0187 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T8 unpooler | B/A 0.6359 [0.6337,0.6469]; P/A 0.4860; AA 0.0311 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T8 router | B/A 0.5623 [0.5543,0.5674]; P/A 0.4567; AA 0.0256 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T16 qkv_gate | B/A 0.7923 [0.7766,0.8120]; P/A 0.7030; AA 0.0179 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T16 o | B/A 0.9758 [0.9508,1.0085]; P/A 0.7877; AA 0.0389 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T16 pooler | B/A 0.8584 [0.8460,0.8852]; P/A 0.7885; AA 0.0188 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T16 unpooler | B/A 0.9928 [0.9785,1.0124]; P/A 0.8911; AA 0.0309 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T16 router | B/A 0.5670 [0.5553,0.5785]; P/A 0.4581; AA 0.0255 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T32 qkv_gate | B/A 0.9911 [0.9788,1.0023]; P/A 0.8904; AA 0.0244 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T32 o | B/A 0.9829 [0.9586,0.9950]; P/A 0.9815; AA 0.0244 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T32 pooler | B/A 0.8636 [0.8398,0.8795]; P/A 0.7893; AA 0.0259 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T32 unpooler | B/A 1.0132 [0.9798,1.0865]; P/A 0.9088; AA 0.0413 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T32 router | B/A 0.5607 [0.5482,0.5687]; P/A 0.4561; AA 0.0232 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T128 qkv_gate | B/A 0.7711 [0.7564,0.7872]; P/A 0.6685; AA 0.0110 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T128 o | B/A 0.9822 [0.9750,0.9872]; P/A 0.9789; AA 0.0253 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T128 pooler | B/A 1.0006 [0.9668,1.0143]; P/A 0.9994; AA 0.0283 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T128 unpooler | B/A 0.9883 [0.9809,1.0084]; P/A 0.9933; AA 0.0184 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T128 router | B/A 0.6800 [0.6708,0.6880]; P/A 0.5700; AA 0.0171 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T256 qkv_gate | B/A 0.9996 [0.9903,1.0041]; P/A 0.9963; AA 0.0145 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T256 o | B/A 0.9923 [0.9431,1.0207]; P/A 0.9932; AA 0.0520 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T256 pooler | B/A 1.0000 [0.9569,1.0189]; P/A 0.9948; AA 0.0623 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T256 unpooler | B/A 0.9961 [0.9851,1.0039]; P/A 0.9954; AA 0.0188 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 output reducer T256 router | B/A 0.6610 [0.6526,0.6711]; P/A 0.5622; AA 0.0253 | One Triton output reducer vs torch sum/cast/add; identical S15 producer | FP64/probe gates pass. Promote component adapter only; production layer already fuses its consumers. | 4e87b1d44d334104 |
| P12 original projection adapter | SGLang/ours .98059 [.96588,.99530]; A:A unavailable | Separate torch sum/cast/router-add output conversion | Superseded by measured one-reducer adapter 08cc94050a4e4564; not a layer regression. | d6ad694a27954c80 |
| P10 fused pool whole-layer A2 | layer speedup 0.9994531 [0.9977760,1.0011301] | NF original12-bit WR1/WK8/S1/ST1 replaces only pool_router at selected T; gate concurrency and fused route/norm retained | A:A independently captured; CI spans1 | 818bd6da9a4b4123 |
| P10 fused pool whole-layer B816 | layer speedup 1.0036173 [1.0019045,1.0053301] | NF original12-bit WR1/WK8/S1/ST1 replaces only pool_router at selected T; gate concurrency and fused route/norm retained | Lower95>1, 768/768 independent TASK checks. C1816 nominated to P4; rebase and official verification required. Outer unchanged S15 host prepare separately failed router autotune. | 818bd6da9a4b4123 |
| P10 fused pool whole-layer C1816 | layer speedup 1.0118260 [1.0103668,1.0132851] | NF original12-bit WR1/WK8/S1/ST1 replaces only pool_router at selected T; gate concurrency and fused route/norm retained | Lower95>1, 768/768 independent TASK checks. C1816 nominated to P4; rebase and official verification required. Outer unchanged S15 host prepare separately failed router autotune. | 818bd6da9a4b4123 |
<!-- P10/P12 measurements end -->

### P10 leader-gap decomposition and exact coding at every dense site

Current deliveries: `problems/muse_bf16_proj/dev/dense_all/P4_HANDOFF.json` has the exact12
T1/sliding gate winner; `dev/large_split/P4_HANDOFF.json` has **no new G0 winner**, two rejected
large-shape rebuilds and an unpromoted optional native-layout arm. The private leader's 1.36125
component advantage (`08cc9405`) is aggregate, not a measured layer advantage or proof of its encoding.
It still has no new same-ABI paired residual measurement; dividing it by a layer R mixes workloads.

`2015fc90abf94e14` decomposes 30 component shapes (8 seeds,A:A): already-coded banks1.50–1.57 B/W,
raw BF16 2, original FP32 router4 (BF16 hi+lo). These are logical bytes, excluding repeated CTA requests
and activation/output traffic; physical HBM counters are unmeasured. Independent checksum streaming
fits4.614 TB/s marginal +2.876 us intercept (max end-to-end4.203), versus judge best4.58 and TASK4.8.
Producer useful rates are0.84–3.62 TB/s. NOPs fit0.519 us/additional launch, first0.944; the stream
intercept is not all launch overhead. Component casts cost1.22–1.79 us and mostly disappear at the
layer's FP32 consumers. `646822ab3d164be8` closes O1 last-CTA norm fusion (1.60–2.95 time ratio)
and20 large-token raw reuse/split schedules (1.10–3.73 with actual neighbours).

`2b45988c88fb42a5` selected exact12 S4/WK4/ST1 QKV: M8full.905404[.897972,.912897],
M16full.931355[.922068,.940737],M16QKV3.917619[.910580,.924713],16seeds/24arms,TASK+A:A.
P4 S19 retained T8; its layer evidence rejected T16. The old three-point handoff is superseded by
that G0 decision. `cc06d34c38284ff2` covers all36 qkv/qkv3/gate/O/pool_router/unpooler cells at
T1/8/16/32/128/256,8seeds8cold,one-hot probes and original FP32 router. All arm CIs are in
`dense_all/ratios.csv`; actual neighbours are retained. Register decoding removes shared decoded-weight
staging/DSM output reduction,98–196regs,zero spills, but active large shapes still lose1.272–2.123.
All-bank GPU prepare adds24.520s (24.404compile); banks are shared across K splits.

`e7b0d5786d174c26` exact raw/coded register-A factorial overlaps two A-register sets with WGMMA,
improving matched code schedules1.3–6.4%, but all active coded large shapes lose. Coded/raw ratios
1.0015–1.183 atT128 and.9965–1.0182 atT256 show the same loss largely exists without decoding.
`ff32499a47704733` K128 groups/adjacent-token tiles improve matched schedules5–19%, still losing
at every active large site,122–196regs and zero spills. These close those RS mappings, not coding.

`e7a553de66604682`: frozen S19,8freshseeds576TASK. Broad code R1.000014[.997643,1.002390],
early-load.999654[.997462,1.001851], QKV+gate single-launch merge.989549[.986697,.992409],
composition.989779[.986745,.992823]. No promotion; a launch saved does not compensate for lost
gate/attention overlap. `e364381789e8445e` confirms fixed T1/sliding-only gate on16disjointseeds,
freshweights/router,576TASK: G0 R1.004405[1.003003,1.005808], A:A.999944[.998391,1.001499].
Prepare170.454s,added9.192s (9.156compile,GPUbank3.525ms). Gate stream and fused sigmoid/output
are retained. This patch remains with P4 for its composition and official decode gate.

**Physical split status: unmeasured.** The `9ab81a3f` observer/launch/sync/checksum/boundary
costs prevent locating S19 physical projection time. Its raw-schedule ablations are archived diagnostics,
not evidence that activation, tensor scheduling or epilogue terms are small. `dev/native_boundary/`
sizes small-M producer excess and native large issue/split-consumer ceilings on S19 itself before building.

`9ab81a3f3b054b92` retains eight-site raw-register synthetic interventions and original TASK checks.
Its stripped-twin residual includes observer, launch, synchronization, checksum and boundary costs;
S19's physical split is **unmeasured**. Neither that residual nor cache-only warming identifies the
size of the native activation, tensor or epilogue term. Archived arm samples/CIs remain in
`large_split/{oracle_split.md,joint_results.json}`; their physical interpretation is superseded by
this limitation and the S19-native one-boundary work below.

**Built shared/shared WGMMA and K-major banks:** `9ab81a3f` tests24 SS schedules/site, same weights
shared across token warpgroups, direct FP32 partial writes, K64/K128, split count and early weight
issue;58–90regs/no spills. Best actual-neighbour stage ratios remain1.112–1.637. `6a7133f514cf4068`
tests42arms/site with8freshseeds, deeper queues, uneven disjoint K splits and exact GPU K-major banks.
All24 same-geometry packed/row-major producer comparisons improve: ratios.886129–.985281,
every upper95%<1. Best packed/S19 actual-neighbour ratios T128 QKV/O/pool/unpool are
1.02069/1.20244/1.00377/1.02296; T2561.11268/1.28889/1.28491/1.21420. Per-shape/arm CIs:
`large_split/{stage_ratios.md,all_stage_ratios.csv}`. Layout causality inside a slower kernel is not a
native S19 win. Round trips verify every bank bitwise; original FP32 router random/column probes pass.

**Held-out whole layer, `7f110a377fb843a7`:** frozen S19,16disjointseeds,freshweights/router,
960publishedlayerTASK +48strict native-pack projection/router checks. The third arm preserves native
S19 tile/stages/warps/S3/CHAIN at T128 QKV/pool, changing only physical weight layout/address.
Six-load paired Student-t15 G0 R values: SS **.971503[.970251,.972755]**, packed SS
**.977755[.976731,.978780]**, native pack **.999817[.999058,1.000576]**; A:A
.999629[.998791,1.000469]. No new promotion. Native verify16 alone wins1.004128[1.002075,1.006184],
which cannot override G0. Native T256 code is unchanged; its movement receives no mechanism attribution.
Combined prepare144.599s; additions SS2.813s, packedSS2.820s, native.315573s (GPU banks.184954ms).
`large_split/np_patch.py` is only an optional unpromoted arm in a planned P4 batch; isolated NP-only
prepare and newer compositions are unmeasured. Normal/forced local official screens each pass13TASK,
12graphs/static; Hopper arithmetic uses explicit local stand-ins and actual native H200 checks above.

Failed receipts are preserved: `017b445a` timed out without rows; code hazards were diagnosed but exact
blocked instruction unobserved. `03ad7fd4` completes native checks but its result packet overruns35B;
neither supplies performance evidence. Final packet retains every timing sample and checks actual wrapper
size. No new private-leader residual is identified. The raw-schedule physical split remains unmeasured;
it is not evidence the activation, tensor or epilogue terms are small.

**S19 native one-boundary sizing and build, `dev/native_boundary/`:** `e70024b150b942fa`
measures frozen S19 with its real kernels/PDL, CUPTI metadata-only graph labels,8freshseeds and
16symmetric A:A timing rounds. Every primary round is retained; profiling is separate and adds no
GPU observer kernel/event node. Actual selected coded-bank bytes are recorded per site. Target stream
=2.876us+bytes/4.61419582TB/s. Residency includes overlapping PDL waits: O16F103.818us residence
has6.076us observed completion contribution; gate32F148.398us has2.657us. Subtracting stream time
from residence would invent independent savings. Exposed-contribution scenarios, not physical bounds:

| Term | Removed us per weighted layer, d1/d16/d32/v1/v16/v32 | Six-load R scenario [sampling95%CI] |
|---|---|---|
| Small-M producer above actual coded-byte target |2.910 /13.278 /8.393 /2.927 /0 /0|1.032884 [1.030488,1.035286]|
| Large native producer surplus +complete consumer removal |0 /0 /0 /0 /26.517 /52.635|1.038411 [1.037944,1.038878]|

The large scenario deliberately grants full b1/route/b2 removal; split-only savings are unknown at this
ranking step. Producer-only modeled surplus is12.187/32.504us at v16/v32; complete consumers add
14.330/20.131us. Scalar throughput, activation/output traffic and added prefetch overlap are unpriced;
the old stream-fit intercept is a target model, not a proven lower bound on each smaller bank. Full
site bytes/residence/exposed CIs are in `native_boundary/SIZING.md`. Model sampling CIs do not establish
a counterfactual gain. This ranks the larger upper scenario for the first one-boundary oracle.

`e4030204ff2c4e2f` tests that term on8disjoint inputseeds and freshweights/router,16palindromic
rounds,320publishedlayerTASK. No inserted timed device nodes: P_READY returns same-seed real-S19
producer partials; SINGLE_READY deletes the same producers and pre-sums toS1 before timing; C_READY
returns native consumer results; BOTH_READY deletes both. Remaining native kernels/PDL are unchanged.
Full producer deletion saves59.198[58.227,60.169]us at v16 and81.685[80.823,82.548]us at v32;
complete-consumer deletion saves15.073[14.512,15.633]/22.632[21.893,23.372]us. Split-only saves
**3.174960[2.435293,3.914626]us** at v16 and**−.051371[−.660838,.558096]us** at v32 (alreadyS1).
Six-load R: P_READY1.086449[1.084538,1.088362], SINGLE1.089104[1.087265,1.090945],
C_READY1.019571[1.018237,1.020907], BOTH1.117580[1.115265,1.119899], split-only
**1.002444[1.001132,1.003757]**. CIs use central `promotion.interval`, max(primary,A:A)SE;
no A:A mean correction. These are deletion/dependency bounds, not candidates or exclusive wait cycles.
Every arm/shape CI and retained timing round is in `native_boundary/{ORACLE.md,oracle_results.json}`.

Built against the native producer term: `c87c2d8aa64b4805` compares24 tile/stage/split dispatches at
all8large sites,8newinputseeds/freshweights/router,16symmetric rounds. Device body is S19's unchanged
`_vdsk_gemm`; only dispatch parameters change, and normal neighbours/PDL remain. No weight bank added.
All448layer +480strict original projection/router TASK checks pass. Best per-site changed-load layer
time ratios (candidate/S19; larger primary/A:A SE) are:

| T | QKV | O | Pool/router | Unpooler |
|---|---|---|---|---|
|128|1.021124[1.019707,1.022543]|1.001106[.999258,1.002958]|1.014280[1.012170,1.016393]|1.002547[1.000578,1.004519]|
|256|1.016783[1.015638,1.017930]|1.034543[1.033110,1.035978]|1.042884[1.041437,1.044333]|1.065163[1.063502,1.066826]|

All24 point estimates lose; closest O128 is inconclusive, not a proven regression. No positive full-layer
point is selected, no new held-out G0 or P4 winner claimed. This closes only these24 plans. All arm CIs
and exact dispatches: `native_boundary/{TUNING.md,tune_results.json,native_patch.py}`. P4 receives negative
results in `native_boundary/P4_HANDOFF.json`; earlier unpromoted native-layout arm and exact12 gate
remain unchanged. Selected-portfolio prepare is unmeasured because no new portfolio is selected.
**Public projection port ranking, `dev/public_ports/`:** native `c74d2a26699743d0` covers all36
T/site cells against S19's actual selected coded dispatch,8fresh seeds,4symmetric rounds per arm;
all individual rounds retained.3604projection +612strict FP32 router +192layer TASK checks pass.
Operatorstack MIT6d9b7765 was cloned into a separate external empty directory and fully reviewed;
only reviewed pure kernels were copied, no donor install/build/test/driver executed. Raw swap-AB/TMA,
PDL twins, an exact coded adaptation and unshipped NF v18 dispatches are compared. Norm/route/gate
neighbours are paid. Every arm CI is in [PORTS.md](../problems/muse_bf16_proj/dev/public_ports/PORTS.md)
and retained samples in `results.json`. Inactive gate/QKV rows are diagnostic; four unpublished NF
large gate/QKV3 shapes are unmeasured, rather than losses.

Candidate/S19 time ratios [95% CI], with larger primary/A:A SE:

| T/site | Port | Ratio |
|---|---|---|
|1 QKV3|raw GEMV|.954440 [.935424,.973843]|
|1 gate/apply|PDL GEMV +paid apply|.812020 [.795931,.828433]|
|1 unpooler slice +norm|PDL GEMV|.934922 [.917315,.952867]|
|16 QKV/gate|unshipped NF|.956081 [.938212,.974289]|
|32 gate/apply|unshipped NF|.978240 [.961561,.995208]|
|32 pool/router +route|PDL tiled TMA|.950465 [.944382,.956588]|
|256 QKV/gate|unshipped NF|.960223 [.943770,.976963]|
|256 O +norm|library path in donor|.977389 [.965936,.988978]|
|256 unpooler +norm|unshipped NF|.989862 [.984002,.995758]|

These nine points rank first; six other positive means with CIs spanning1 also enter the next batch.
No stage CI is a new promotion gate. QKV3T16's .678863 [.223709,2.060059] includes an unexplained
1.975712ms A round; every sample is retained, with no confirmed gain inferred. S21 already has the
exact12 sliding T1 gate, so the S19 gate stage result cannot establish an S21 improvement.
Pool/routerT16 raw1.047549 [1.027654,1.067829], PDL1.0438 [1.0237,1.0642] lose; T32 raw.9689
[.9555,.9824] wins. The new exact payload +sparse escapes preserves every BF16 pattern and all native
TASK checks, but this Triton swap-AB adaptation loses badly (poolT16/T32 ~15.94x/17.24x). Its
representative sm90 cross-compile uses255regs and1016B stack; this is supporting resource evidence,
not an exclusive physical decomposition or a closure of lossless coding.

Preparation on this H200 lab: baseline137.812s, all added banks/NF variants57.585s. GPU exact-code
packs O/pool/QKV/unpool take .002482/.010882/.005759/.009381s; raw tile permutations total<.0004s.
Most added cost is compiling all unshipped NF families. Selected S21 prepare, including warmup, is
still unmeasured. Exact MIT-noticed patches and all15 point specs are in `public_ports/P4_HANDOFF.json`.

Built the next small-M test with S19's existing exact12 NF reader: WK4/8, ST1/2, first weights before
PDL activation wait; and donor tiled TMA BK128/256 ST2/3 to admit more CTAs, with matched early-issue
controls. Local full S21 rehearsal passes848layer+640strict projection/router TASK; all8changed
sm90 TMA schedules compile at61–98KB shared and PTX verifies weights-before-wait only in early arms.
Twelve NF sm90a kernels also compile with72–128regs/zero spills and intended PTX load/wait order.
Exact P4 patch normal/forced screens pass13TASK/12graphs/6static each.
The native follow-up is **unmeasured**: all five hosts exhausted0scored+0checks after the port run.
Sender3347492 (`9654f02c`) automatically waits the real10-27 09:01KST reset. It freezes S21eacc087b
and batches all15port points individually/portfolio plus10pool variants on8disjoint fresh seeds and
8symmetric rounds. No second duplicate queued. `queue.json`, `whole_manifest.json`, and
`whole_validation.json` preserve the exact ready work. Read its receipt with `whole_read.py`, then
P4 must apply the surviving portfolio on S21/S22 and pass its own held-out six-load G0. No new G0
winner, native pool-prefetch result, selected prepare, or private leader residual is claimed.

Transport failures `f304369a` (222bytes over) and `24b58b2d` (156bytes over) are unmeasured, not
kernel losses. Successful repeats keep all scoped observations/arms/seeds/rounds. `send.py` calls the
normal h200_lab sender with a process-local14k reservation:13744packet+256markers+2000suffix fits
16k, tested with the actual wrapper and injected suffix. No shared tool/rule/timing change. The five
canonical source/receipt pins and all negative evidence are retained in `native_boundary/`.

## Glue and routing

### P13: boundary compute / route deletion on S19 (2026-10-26)

Exact S19 `62ea5bf4`; corrected **611341aaf8454a24**, projection-host import,
8fresh160001021+104729*i seeds ×10complete symmetric cold rounds, all12S/F.
Both layer types use `dist=soft`; exact public ref_front biased top8 chooses
the busiest shard separately per fixture; all96 local_start values retained.
480TASK pass; all rounds and576NVML snapshots losslessly archived, no trimming.
R uses46S+16F per load and unchanged promotion.py paired-seed Student-t interval.

| Oracle | d1 | d16 | d32 | v1 | v16 | v32 | Six-load R [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| AA | 1.01201 | 1.00028 | 0.99803 | 0.99971 | 1.00080 | 0.99943 | 1.001700 [1.000617,1.002783] |
| ROUTE | 1.04313 | 1.01497 | 1.01042 | 1.02392 | 1.01793 | 1.01203 | 1.020342 [1.018712,1.021974] |
| BOUND | 1.00443 | 1.03659 | 1.03761 | 1.01989 | 1.03573 | 1.04513 | 1.029805 [1.028333,1.031279] |
| BOTH | 1.04794 | 1.04997 | 1.04277 | 1.05493 | 1.05379 | 1.05823 | 1.051259 [1.049754,1.052766] |

ROUTE removes pooled norm/top8 plus standalone large-M expert metadata;
BOUND removes b1+b2; BOTH removes both, while retaining every producer/expert
compute. These are deliberately excluded-work oracles, never valid candidates.
A:A's0.17% shift is retained (CI excludes1); using AA as reference still gives
BOTH1.049475[1.048121,1.050832]. No centering/subtraction or gate change.
BOTH LB1.049754 clears the predeclared~1.01 build gate; no attainable5.13% promise.

Built own route fusion first (ROUTE LB1.018712): identical coded raw pool/router
producer +last-CTA norm/top8 atT1/8/16; row route +last-CTA exact S19 expert
metadata atT128/256; no spinning, all writers fence, cyclic arrivals reset
within the call. B1/b2 andT32 remain S19. The large-M neighboring library
GEMMs have no matched public epilogue; BOUND remains an open implementation.
Reviewed operatorstack6d9b7765 MIT and lee101601c6541 Apache-2.0 in separate
empty directories; no donor scripts executed/code copied. `donor_review.json`.

Corrected native pair **fec0b6726aa0413d**, MoE-host import (own unsubmitted
projection-quota waiter3276205 cancelled before exact payload migration):
8fixed fresh170001021+104729*i seeds, same10-round all-sample protocol;
480primary+72public/nearby TASK pass, prepare estimate131.65s, private153.35s.
Combined R**.957134[.955297,.958973]**, A:A1.000650[.999705,1.001595]: **fails G0**.
SMALL .963569[.961729,.965413], LARGE .992416[.991479,.993353] both lose;
only combined has its own same-source A:A, so separate arms are diagnostics.
No winner/official submission. P4 negative exact patches in
[dev/p13_boundary/P4_HANDOFF.json](../problems/muse_decode_layer/dev/p13_boundary/P4_HANDOFF.json).

Mechanism checked vs inferred: SMALL adds27.65us d16 /10.78us v1 in the whole
layer; source serializes16/8 row epilogues in one last CTA instead of8CTAs/token.
sm90a compiler proves0stack/spills after removing dynamic top8 indexing;
metadata64regs, poolMT1/MT2=80/94regs (MT2 was80 before). This schedule and
occupancy cost fits the losses; it is not a measured per-instruction partition.
LARGE adds9.72/3.84us v16/v32; publication fences/arrival and serial metadata
remain, individual costs unpartitioned. Close these two route recipes only,
not the measured BOUND opportunity or all possible route fusions.

Evidence scope correction: **752af6d0c91943f8** is unmeasured (packet11799B >11744B);
process-local14k reservation fixes transport, no shared-tool change.
**14f5558bcec443ea** and **434fdf8f627d41ff** are valid fixed-shard/coherent
diagnostics only: the initial helper defaulted local_start0/full coherent.
Their score interpretation is withdrawn; `fixed_shard/SCOPE.json` and all
immutable receipts retained. Corrected repeats use new seeds and both-soft,
busiest-shard fixtures; no sample was selected away. Local admission uses
explicit S19 Hopper stand-ins, current480+72checks and2sm90a units compile;
native fixtures/gates untouched. Official candidate TASK and independent
prepare are unmeasured because no patch qualifies. No RULINGS entry.

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
| d1 glue chain, 32 seeds | W S/F 0.98648/1.00369; D full 0.98549; A:A S/F **0.98549±0.00313 / 1.01603±0.00539** (95% mean CI) | Gates pass; the earlier A:A veto is dropped under the GOAL G0 gate. Ten changed d1 shape pieces go to P4 by point estimate; C4/hints still have no sub-1 point. C4 observed 1.03119/1.02371, hints 1.00879/1.01158 | Same-capture balanced AB/BA confirmation of preload ownership; source of A:A bias is not isolated | `5a95984a640d410e` |
| Full-row moment/exact, serial, route CTA and chain at d16/d32/v16/v32; 24 seeds | M **1.07236/1.05352/1.01380/1.00461**; E **1.07590/1.05995/1.01794/1.00843**; R **1.05884/1.07432/1.00838/1.00409**. A:A **0.99813/1.00069/1.00392/0.99991** | No changed recipe/shape has a sub-1 point under the current gate. d16S ready b1 A/M/E 7.168/13.712/13.712 us; route A/R 8.688/16.992 us. The old membership-difference veto is superseded; routing near ties are judged only by TASK | Other CTA geometries/producer ownership and held-out TASK/G0 confirmation; no family-wide closure | `9f6849fcc12a448d` |
| Decode per-tile unpool publication + exact b2 | d16/d32 **1.08951/1.07456**, same A:A as above | Close this producer/consumer recipe: sampled b2 completion tail 2.753→15.488 / 3.167→14.304 us. Verify P is an unchanged duplicate control, not a publication test | Other readiness protocols, faster current producers, and finish ownership; no general publication/fusion closure | `9f6849fcc12a448d` |

### GOAL G0 gate point nominations to P4

The former per-load A:A veto is superseded. Ratios are candidate/control latency, SE is across32 paired seeds; these are selection points, not held-out promotion. Full seed IDs and exact S15 patch hashes are in `problems/muse_bf16_proj/dev/d1_chain/point_nominations.json`. P4 composes them and confirms the paired six-load R lower95% bound>1 using TASK correctness only. A:A remains diagnostic (d1S .98549, d1F1.01603); its mean is not subtracted. D d1S and rebuild-P verify are unchanged duplicate controls, so they contain no piece to integrate.

| Recipe / shape | Ratio (SE); seeds | Mechanism / changed verdict | Leaves open | Receipt |
|---|---|---|---|---|
| W all immutable glue preloads, d1S | .986479 (.001738);32 | Nominate to P4; formerly withheld for A:A | Held-out composed six-load effect on S15 | `5a95984a640d410e` |
| L final prew late, d1S/F | .991532 (.001940) / .995726 (.001788);32 | Nominate both shapes; role ownership differs from W | Held-out six-load confirmation and W/L choice | `5a95984a640d410e` |
| S early original b1 residual, d1S/F | .995325 (.001614) / .999815 (.002240);32 | Nominate both point estimates, including the small full point | Held-out composition; mean point alone does not establish a gain | `5a95984a640d410e` |
| W1 b1-only preload, d1S | .987399 (.001864);32 | Nominate role-only row alongside W | Whether W1 composes better than W with attention changes | `5a95984a640d410e` |
| WR route-only preload, d1S/F | .998331 (.001476) / .998975 (.001591);32 | Nominate both small point wins | Held-out six-load effect and interaction with router producer | `5a95984a640d410e` |
| W2 b2-only preload, d1S | .997874 (.001613);32 | Nominate role-only row | Interaction with new residual-tail reuse | `5a95984a640d410e` |
| D prior full-only preload, d1F | .985488 (.002062);32 | Nominate to P4; d1S is duplicate control | Held-out S15 composition; no historical score transfer assumed | `5a95984a640d410e` |

### S15 MoE shell point nominations

| Recipe | Ratio vs control (SE) | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| S15 MoE P original-ownership dependent finish, v16S/F + v32S/F | .993326(SE.000755) / .996274(.000711) / .996526(.000268) / .996821(.000364);32 seeds. A:A .999591/1.001177/.999730/1.000158 | Nominate all four points to P4. Typed dependencies from all three W2 branches; inline PTX keeps SS/Y after wait. Sampled P starts near heavy/light completion; small W2 end was not retained by a label-filter bug, so exact three-branch tail is unknown | Held-out composed six-load score/TASK confirmation, new P3b producer interaction, and small-W2 frontier | `1a7bbd9e33c04466` |
| S15 MoE F4 v16S; F8 v16S/F; E v16S | .999530(SE.000831); .998367(.000591)/.999197(.000551); .999100(.000674);32 seeds | Nominate every sub-1 point, no A:A veto. Warp-private finish removes CTA hand-off; E changes only small W2 entry release | Point gains are selection observations; held-out six-load/TASK confirmation and composition with P | `1a7bbd9e33c04466` |

### S15 residual-tail reuse, total T<=8

All640 copied public TASK checks pass,32 paired seeds, A:A capture signatures identical; A:A ratios d1S/d1F/v1S/v1F **1.011209/.993489/.997094/1.009460** (SE .003787/.001628/.002187/.002267). A:A is a noise diagnostic, not a per-shape veto. Exact patches and seeds: `problems/muse_bf16_proj/dev/residual_tail/point_nominations.json`. Selection points below require P4's held-out six-load G0 gate; none is declared promoted here.

| Recipe | Ratio vs control (SE), d1S/d1F/v1S/v1F | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| V: b1 writes b2's final7168 residual elements |1.013104(.003155)/.996049(.001739)/1.033689(.002733)/1.038322(.001787) | Exact identity confirmed; nominate d1F only. Verify store transfer loses3.4-3.8%; b1 also lengthens | Other tail ownership/scheduling and held-out d1F contribution; no general identity/fusion closure | `e841e2e8b1d04f0c` |
| N: b1 writes seven squared-sum partials; b2 reduces first1024 |1.007888(.003012)/.985986(.002102)/.997537(.002102)/1.001321(.002010) | Nominate d1F/v1S points. Full8192 denominator/all outputs retained; N b1/b2 use96 regs vs57. Sampled residence changes do not establish critical causality | Disjoint-seed composition and tail-CTA specialization; A:A drift and overlap effects remain | `e841e2e8b1d04f0c` |
| C: V+N combined |1.012938(.002364)/.990038(.001966)/1.004485(.002453)/1.008552(.002053) | Nominate d1F; combined kernel111 regs, no advantage over stats-only N in any point | Held-out six-load composition; better split of prefix/tail work and producer overlap | `e841e2e8b1d04f0c` |

### Stats-only repeat and dedicated tail path

| Recipe | Ratio vs control (SE) | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| N repeat on32 disjoint seeds |1.016186(.003004)/1.004723(.001935)/1.002925(.002758)/.997095(.001811), d1S/d1F/v1S/v1F; A:A1.019137/1.003947/1.004135/1.022328 | All512 batch TASK checks pass; original d1F/v1S point wins do not repeat. Nominate the new v1F point to P4; other earlier points remain selection data | Held-out six-load composition and directional capture effects; no per-shape A:A veto or family-wide closure | `11d531dcb81d4b7f` |
| T: dedicated N b2 tail path, two ordered immutable preloads |1.000877(.003282)/1.003423(.001393)/1.038238(.002044)/.994935(.002209), same shapes/control | Nominate v1F to P4. N b1 handle unchanged; b2 N96/T98 registers. Sampled T b2 critical increments decrease.064/.160/.128/.128us versus N | v1S slowdown is upstream in the representative trace: pool/router+1.888 and MoE+2.112us before b2 starts. Allocation/capture/clock or graph effects unisolated; exact v1F patch e5414dd7 onto S15 | `11d531dcb81d4b7f` |

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

Score-gate audit: [63 rows / 90 primary receipts](../problems/muse_decode_layer/dev/gate_audit/README.md),
with [per-recipe verdicts and retained seed statistics](../problems/muse_decode_layer/dev/gate_audit/rejudgment.json).
`0f74021a` B (weight hints + metadata), M (metadata), and D (prior d1-full glue) now pass the six-load
**timing** gate: R 1.002694 [1.002068, 1.003320], 1.003169 [1.002493, 1.003846], and
1.002841 [1.002001, 1.003683], respectively. These are nominations for current-base TASK validation and
disjoint confirmation; the old lab's host-equivalence checks do not certify every arm against TASK.
The all-register-preload policies still lose: R 0.885121 [0.883630, 0.886615] / 0.887139
[0.885708, 0.888572]. Page-flight and heavy wide/adaptive shape points remain nominations where exact
six-load seed covariance was not retained. A host-only error/ID rejection leaves a recipe unmeasured.

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
| Persistent qkv chain (`commit 9d8785c`) | Host-only rejection at (32, 1) nondeterministically; 0.86% hidden difference and changed expert IDs at (8, 8) | Does not establish a TASK failure; no paired six-load receipt | Reopen with TASK and deterministic replay; timing unmeasured | local; score-gate audit |
| Glue fused into streaming GEMVs (norms, residual, routing) | 2.3-21x cuBLAS (`5535d2f1`); every fused stage, prefix and chain 1.04-11.4x (`479aab30`) | Folding glue into skinny GEMVs costs their parallelism | Glue fused into consumers | `5535d2f1`, `479aab30` |
| Cluster glue at t = 128 / 256 | Every cluster shape 1.02-1.28x of S6's Triton glue | Not isolated | t <= 32, where it wins ("Glue and routing") | `1365a290` |
| Glue small-M fusion, prologue and last-CTA epilogue cells | 1.51-17.63x slower | Implementation bug: the pool grid recomputes each 8192-column norm 576 times per token; last-CTA tails serialize on 1/1/2/4 CTAs at M=1/8/16/32 | Repaired cells (next row) | `80b70f7f` |
| Glue cooperative resident grid: one RMS/row, 8 tails/row, route-once v1 | Pool 1.40-5.67x, tail 1.16-5.91x, all 1.65-9.53x, epi 1.58-9.93x, route-once 1.02-5.53x; 12 cold cases, gates pass | Grid-wide barrier/row suspected, not isolated | Fusion without grid-wide sync | `9eeb39ba` |
| Side-stream L2 prefetch beside the layer | Layer 0.3-16.6% slower (decode 1 sliding 123.1 -> 143.5 us); branch ~34 us | A separate launch can't run beside full-grid kernels (it overlapped attention only, never MoE) | In-kernel prefetch | `d5a90258` |
| In-kernel PDL own-weight prefetch | d1S 115.78→115.58 us, 0.9986; roles 0.9910-1.0094 | 7/8 cases unmeasured: host-only 0.003 gate rejected 0.0031-0.0033; this is not TASK failure | Reopen immutable-weight hints under TASK; six-load R/CI unavailable | `7426e17c` |
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
| Force all intended PDL preloads before wait | Register policies R 0.885121/0.887139, upper95 <0.889; B/M/D now have lower95 >1 (audit above) | Registers 48→150, LDGSTS 40→8 closes the register recipe only | Nominate hints+metadata, metadata and d1-full glue; TASK then current-base disjoint G0 confirmation | `8db42132`,`0f74021a` |
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

### S17 cross-stage continuity (P7; S15 bound)

S15 receipt `7eaa262a30ec483e` contains CUPTI activity, not DRAM counters. Therefore HBM-idle
time is unidentifiable. [P7 BOUND.md](../problems/muse_bf16_proj/dev/stream_continuity/BOUND.md)
records every critical interval and each boundary's logical bytes/time plus exact GPU-idle gaps.
Times below are `(46 sliding +16 full)/62` us/layer at each load. The scenario ceiling retains the
initial qkv ramp, permits one fitted2.36us ramp per weight/KV boundary, excludes glue intercepts,
combines concurrent MoE branches and adds only actual GPU idle. Busy DAG gaps remain charged to
their running owner. It is not a physical HBM-utilization bound or demonstrated saving.

| Load | GPU idle us | Continuity ceiling us | Retained materialization HBM-equivalent us |
|---|---:|---:|---:|
| d1 | 0.00 | 13.45 | 0.15 |
| d16 | 0.13 | 13.59 | 2.10 |
| d32 | 0.48 | 12.89 | 3.30 |
| v1 | 0.35 | 12.15 | 1.08 |
| v16 | 5.45 | 17.25 | 8.19 |
| v32 | 5.86 | 17.66 | 13.80 |

Joint continuity model R1.08557 closes32.8% of the initial GOAL snapshot gap (1.83789→2.3601).
The retained-operand scenario removes one store/read per interstage output at4.58TB/s, excluding
returned b2 outputs and one required route-ID store: joint HBM-equivalent R1.01602. Existing L2 hits
reduce HBM savings; on-chip retention may also save L2 issue or repeated activation/page reads.
Those are unmeasured, not closed by this small byte-service estimate. The171/403MB MoE activation
reread estimates belong to internal multiplicity, not an additive interstage cold-HBM saving.

The interrupted S15+C arms were rebuilt on frozen S17 `fea71760`. Prefetch follows the consumer's
actual NF/coded/LT/raw banks, including `p10_pool_raw`; the old original-BF16-bank warming was removed.
S17 NORM/PAIR/CHUNK4 arithmetic and dispatch remain intact. ptxas initially sank intended O loads
past the dependency wait; the existing NF order-point branch now keeps all6 initial128-bit weight
vectors before ACQBULK at O1/O8. Nine changed sm90a sources compile without spills. Normal and
forced local batches each pass1728 TASK checks plus13 host tests/12 graph shapes. The forced
screen covers the same Python ABI before the final order-point-only native change; final normal
and sm90a/SASS audits cover that change.

Whole-decode receipt `a405c8b463054e6c` used a **prefill-host check** because decode had spent20/20
checks, preserving the H200, synthetic decode fixtures, public TASK checker and paired six-load
measurement. The outer prefill score is irrelevant.16 frozen held-out seeds×8 rotated cold rounds,
12 shapes,1728 TASK checks all pass; A:A fingerprints agree. R is incumbent/candidate six-load
geomean; two-sided Student t95 on log R across16 paired seeds (df15), no A:A correction/per-load veto.

| Arm | Mechanism | R | 95% CI | Verdict |
|---|---|---:|---|---|
| A2 | identical control | 0.999820 | [0.998789, 1.000852] | diagnostic |
| E | release after final TMA issue | 0.994282 | [0.988018, 1.000585] | fails G0; CI spans1 |
| P | E + late4MiB unpool-bank prefetch | 0.989306 | [0.987589, 0.991027] | loss |
| W | O1/O8 NF weight loads before wait | 1.003215 | [1.001964, 1.004468] | G0 pass |
| F | 4MiB ordinary reads in PDL intermediaries | 0.382012 | [0.380140, 0.383893] | loss |
| B | 4MiB L2 hints in PDL intermediaries | 0.914259 | [0.909898, 0.918640] | loss |
| L | 16MiB bulk L2 hints | 0.615109 | [0.613363, 0.616860] | loss |
| X | frozen P+W+B composition | 0.914508 | [0.912607, 0.916412] | loss |

**Handed P4:** W whole-arm clears G0, including its losing d1F/v1 pieces. Its per-load R is
1.020060/1.000193/1.000130/.998878/.999620/1.000577. Only O1/O8 launch functions are swapped;
S17's packed allocations and every other role function are retained. Appending the generic
W transform directly would also alter pool/router preparation, so the production patch builds
only O launchers after unmodified base preparation. A d1S-only W alternative (R1.033081 shape
point) and E d1S/d32S/d32F points are selection nominees, requiring P4's composed held-out G0.
Unchanged-shape point fluctuations are duplicate controls, not pieces to integrate.
The standalone O-only production candidate passes13 TASK/12 graph shapes in each normal/forced screen.
Exact patches/hashes/seeds/CIs: [P4_HANDOFF.json](../problems/muse_bf16_proj/dev/stream_continuity/P4_HANDOFF.json).

The tested W recipe recovers0.32% of layer time vs the8.56% modeled ceiling; the ceiling was not
achieved. P/F/B/L/X close only these prefetch schedules; E's whole-arm interval is inconclusive.
HBM/issue contention and the extra prefetch launches are plausible costs, not isolated causes.
No family-wide continuity or retained-operand conclusion follows. Larger open MoE/attention/
coding levers take priority over another standalone sub1% follow-up; S18 composition remains P4.

### S18 consumer-owned first-stage extension (P7)

**Measured:** prefill import-time check `73d2d80dd74848b7`,16 fresh paired seeds x8
rotated cold rounds,18 arms x12 shapes =3,456 unchanged TASK checks, all pass; A:A
native kernel/resource/stream/DAG fingerprints agree. Frozen S18 `95df42fe`, not
S19; elapsed555.0s. An earlier setup-only failure `724e9b8d` has no timing verdict.

| Consumer boundary | R | 95% CI |
|---|---:|---|
| qkv/qkv3 (Q) | 0.666395 | [0.664339, 0.668457] |
| gate, already covered (G) | 1.001344 | [1.000094, 1.002596] |
| O | 0.970939 | [0.969130, 0.972751] |
| pool_router (P) | 0.613680 | [0.611303, 0.616067] |
| unpooler/slice (U) | 0.710344 | [0.707545, 0.713154] |
| NT1 W13 (N13) | 0.987878 | [0.986632, 0.989126] |
| NT1 W2 landing (N2) | 0.999361 | [0.997670, 1.001056] |
| heavy W13 (H13) | 1.000721 | [0.999565, 1.001879] |
| heavy W2 (H2) | 1.001504 | [1.000124, 1.002887] |
| b1 (B1) | 1.001355 | [1.000064, 1.002647] |
| b2 (B2) | 0.997603 | [0.996006, 0.999203] |
| first KV pages (KV) | 0.995181 | [0.993115, 0.997252] |
| all boundaries (ALL) | 0.382568 | [0.379952, 0.385202] |
| editable composition | 0.991852 | [0.990393, 0.993313] |

H2 heavy-W2 and B1 pass raw G0 (lower CI >1, also after a0.5ppm archive-rounding
guard). G is an unchanged GPU coverage control and also has CI >1; paired H2/G
1.000160[.998620,1.001702] and B1/G1.000011[.999125,1.000897] are inconclusive.
A:A1.000122[.998463,1.001783]. Retain raw gate results and this control evidence;
these are marginal P4 nominees, not a demonstrated causal gain or a S19 verdict.
H2 acts at v16/v32 (v16R1.006394[1.003615,1.009180]); B1 wins d1/v1 but loses v32.
No per-load veto or A:A correction was introduced.

ALL includes native substitutes for opaque projection/attention consumers; their
inner-loop cost dominates. LATE .373568[.371663,.375482]; ALL/LATE1.024093
[1.020797,1.027399] improves that schedule package but remains far below S18.
EDITABLE keeps incumbent cuBLAS/Lt/FA3 and still loses. KV_LATE .994803
[.992684,.996927], KV/KV_LATE1.000380[.998779,1.001983]. These controls prevent
calling the native-reader replacement cost a preload-only result.

**Static/local:**39 exact sm90a CUDA and25 sm90 Triton builds pass,146 acquire
placement witnesses. QKV3's register prologue79->90 would lower residency; its
smem first stage restores79. Native glue consumes its copied smem first group,
with B1/B2 isolated; selected resource ceilings retain occupancy. The original
OL source injected GDC but launched with `<<<>>>`; changed OL enables actual PDL.
Two DSOs also shared a GNU-unique `configured` guard: a native local ABI stand-in
reproduces error1/49,152-byte default, then private guards pass64KiB eager+graph.
Final regular local_eval passes13 TASK cases and12 captures; forced full-arm lab
passes3,456. Small native KV/gate and NT1 W2 already issue early. NT1 W13 addresses
need route IDs; delaying route admission until publication avoids an ALL-chain
spin deadlock. The heavy W13 fork event remains serial. These are code exposure
limits, not counter evidence for bandwidth contention.

**Bound/standing:** retain the S15 scenario R1.08557 (per-load13.45/13.59/12.89/
12.15/17.25/17.66us); it is broader than first-stage scheduling and is not physical
HBM utilization. ALL recovers0% positive share; H2/B1 raw gains recover1.83%/1.65%
of its log bound. No general continuity or P11 retention ceiling follows. Larger
MoE/attention/coding levers stay ahead of further standalone prefetch schedules.
[P4_HANDOFF.json](../problems/muse_bf16_proj/dev/stream_continuity/extend/P4_HANDOFF.json)
contains H2/B1 exact sources and readable patches, all controls/losers and the OL
launch fix; fresh S19 composition belongs to P4. P4's S19 already dropped old W.
[Full result, per-load CIs, evidence and reproduction](../problems/muse_bf16_proj/dev/stream_continuity/extend/README.md).

### P3b / P3a public MoE ports: native and matched ABI (2026-10-26)

Reviewed, isolated external clones: operatorstack MIT `6d9b7765` and Lee Apache-2.0 `601c6541`, MoE v43d. Only reviewed kernels/repackers were copied; no donor driver, install/build script, tuner, test or notebook ran. Complete MIT/Apache texts and operatorstack NOTICE are retained in `dev/moe_public_ports/NOTICES.txt` and embedded sources. No RULINGS change.

Native cohort **`550f0aa88785489f`** versus exact S19 `62ea5bf4`: 598 TASK checks, eight fresh input/router seeds on one fresh original-precision weight bank, four cold rounds in rotated forward/reverse cohorts, every integer-ns round retained; all six A:A CIs contain one. Whole-MoE graphs retain original S19 fusion and branch overlap. Both Lee RS-WGMMA configurations and both published small-M F choices were tested. Time ratios port/S19, 95% paired-seed t7 CIs; best Lee choice shown (unadjusted per-arm CI), all variants in the evidence.

| M | operator | best Lee choice | Lee / S19 | pick |
|---:|---:|---|---:|---|
| 1 | 1.220410 [1.182233,1.258586] | F1 | 1.601007 [1.575827,1.626186] | S19 |
| 8 | 1.203872 [1.138832,1.268912] | F2 | 1.366111 [1.262242,1.469981] | S19 |
| 16 | 1.111078 [1.058164,1.163992] | F2 | 1.351470 [1.293243,1.409697] | S19 |
| 32 | 1.055137 [1.036481,1.073794] | F2 | 1.310762 [1.259718,1.361806] | S19 |
| 128 | 1.058696 [1.020878,1.096513] | X2 | 1.223691 [1.205230,1.242153] | S19 |
| 256 | 1.175861 [1.131331,1.220390] | X2 | 1.187328 [1.126048,1.248607] | S19 |

No whole port wins. The native phase cohort had three points below one at M128: O W13 .952213 [.894279,1.010148], O W2 .911612 [.874737,.948486], Lee X2 finalize .914463 [.870148,.958778]. A contract audit found native O W2 excluded S19's inline norm/SS and Lee finish read BF16 Y/scalar SS instead of FP32 Y/64 SS. These ratios are not transferable phase wins.

Matched-ABI follow-up **`17b91ef9e95841e7`**: 132 TASK checks, eight new input/router seeds and another fresh weight bank, four symmetric cold rounds retained, both A:A CIs contain one. Operator consumes S19 metadata and writes sorted BF16 ACT, FP32 norm-weighted Y and 64 SS parts; all metadata costs are timed. Lee finalize consumes actual S19 FP32 Y/64 SS. Original complete S19 remains the control.

| M | matched O W13 | matched O W2 | matched Lee finish |
|---:|---:|---:|---:|
| 128 | 0.981412 [0.904507,1.058317] | 1.025524 [0.943486,1.107562] | 1.350141 [1.307908,1.392374] |
| 256 | 1.016750 [0.964685,1.068815] | 1.093285 [1.032576,1.153994] | 1.541246 [1.431000,1.651492] |

All eight actual hybrid whole-MoE cells lose (O13/O2/O132/LF at M128: 1.092380/1.136557/1.102665/1.014787; M256: 1.122748/1.153772/1.159132/1.014262; every CI lower bound >1). Operator M128 W13 is an unresolved point; metadata is 8.665 vs 3.420 us, and losing S19's branch overlap costs more. Source mechanism: direct packed-word register loads, bit-permuted BF16x2 decode, shared activations, per-item 8/16/32 rows; not a causal decode-only ablation. Lee native finish's byte/sum advantage disappears at the real ABI. Unchanged S19 finish control points are not donor gains.

**Decision: S19 at every M; no winner patch or G0 nominee to P4.** P3b's tested public-port recipe is closed; no new matched speedup mechanism justifies a >1% structural build. This does not close the MoE ceiling or other kernel designs. Small-M results and every native donor phase time are in the P4/P3a handoff; separate S19 W13/W2/finish denominators at M1–32 do not exist because the body fuses them. Whole-layer G0 remains unmeasured.

`5e45775a66a9416f` is unmeasured: the first official M4 TASK gate exposed a missing generic S19 counter binding in the reduced prepare; restored original `smallm_bind`, expanded local forced glue to M4, reran unchanged primary cohort. Local verification: 50 real eager/graph port cases, 64 forced Hopper-glue calls, five matched-ABI eager/graph fixtures (M33/128/256/257 and skewed128), four dispatched sm90a compiles plus changed FP32 finalizer. Actual X SASS contains HGMMA; F contains HMMA. Native runs completed in 41.1 / 36.6 s including setup.

Evidence: [native ratios](../problems/muse_decode_layer/dev/moe_public_ports/H200.md), [matched ABI](../problems/muse_decode_layer/dev/moe_public_ports/ABI_H200.md), [mechanisms](../problems/muse_decode_layer/dev/moe_public_ports/MECHANISMS.md), [P4 / P3a handoff](../problems/muse_decode_layer/dev/moe_public_ports/P4_HANDOFF.json). Receipts/source snapshots are in `runs/muse_prefill_layer/`; all raw rounds and source identities are in `receipt_data.json` / `abi_receipt_data.json`.

### P3b exact S19 large-M excess ladder: queued, not yet measured (2026-10-26)

P3b is active. Public MoE ports remain closed; the large-M S19 excess is still
unassigned. Read S18 `3330edc2`: its ~15 us gather and 26/56 us sync belong to
added transport paths, and the checksum stream changed granularity/addressing.
Those results do not assign S19's 32.63/67.58 us MoE-core excess at M128/256.

Built a single exact-S19 `62ea5bf4` cohort covering W13 and W2's three concurrent
branches, including P7 small W2. L0 loads, L1 decode/scale, L2 MMA and L3
epilogue retain tiles/grids/issue order. BASE/AA share the cuts' build with
original allocations; on-chip MATH and unique-byte STREAM provide references.
Original padded MATCH, full SHADOW, INIT and instrumented HIST expose
resource/sink/observer costs. Eight fresh layer weight/router/input seeds, four
rotated forward/reverse cold rounds, every integer-ns sample and round telemetry;
expert counts, CTA/SM work histograms and exact last-CTA finish retained. Frozen
metadata avoids atomic route-order changes; real W13 resets each W2 fixture.

**Checked locally, not H200 timings:** 384 full-math TASK comparisons, full wire
cohort, all diagnostic eager/graph paths, two sm90a compiles, 64 actual SASS cut
audits, and lossless independent reader/work-count validation. L0/L1 have no MMA;
L0 no BF16 products; MATH/INIT no global async operand copies. Static spill
bytes are disclosed: original/SHADOW heavy W2 4/4, L3 W13 8/8 and W2 12/16,
HIST 64/56. Controls must qualify any attribution. Final packet 11,154 bytes
under11,744; all1,536 timing integers remain exact. Histogram summaries use
256ns bins and exact last-CTA timestamps, from one final HIST replay per seed.

**Hard quota wait:** all five hosts0scored+0checks. Queued via h200_lab, laneP3b,
decision"assign the large-M MoE excess", priority6. Wrapper `bad55569a98d1c1d`,
payload `38c3c1f24bc1a0a3`; sender3645567 wakes2026-10-27 09:01KST.
This is a queued source identity, not a judge receipt. It replaces unmeasured
waiters4c0e2cd5/ad8b29e1 to improve lossless transport and put BASE/AA in the
same translation units as the cuts at original allocation; no quota spent.
Largest term/CI, term ceiling and ranked build remain pending. No G0 winner or
patch has been handed to P4; no lane closure follows from quota exhaustion.
Existing timeline9c16c850 full-removal scenario is R1.04778; it is not an
achieved gain. P3a's row is unchanged.

[Design, caveats, queue and reproduction](../problems/muse_decode_layer/dev/moe_excess/README.md).
`read.py` will emit native us/paired t7 CIs and conditional contrasts, verify
all round/telemetry/work/TASK evidence, and report capped six-load scenarios.
Fresh whole-layer candidate confirmation still requires tools/promotion.py.
A local inotify/pidfd receipt reader is attached; it will materialize validated
raw/summary/ladder artifacts on completion, without judge calls or promotion.
