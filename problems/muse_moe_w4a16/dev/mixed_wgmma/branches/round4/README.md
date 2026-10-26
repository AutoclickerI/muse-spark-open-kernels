# Heavy TMA / decode / RS-WGMMA pipeline

Current candidate: **WS_PIPE**, payload SHA256 **80c5d1c8**, CUDA **a43783e7**.
Rebuild with `python3 problems/muse_moe_w4a16/dev/mixed_wgmma/branches/round4/build.py`.
Full identities are in [manifest.json](manifest.json); generated payload/CUDA/adapters are ignored.
The builder appends the pipeline to the exact CTA2 control prefix **5aff055b** and retains
S14 **f1c77ce6** as the other control. This is a lab, with no production promotion before H200.

## Design

Each 384-thread CTA statically owns an expert and two adjacent 64-output tiles, processing all
of that expert's tokens up to256 with register accumulators. Three roles advance independently:

- WG0: one TMA producer warp plus three decode warps. The producer streams9216B lossless packed
  code/scale packets into a two-slot raw ring, and BF16 X into a two-slot swizzled ring. The decode
  warps use the unchanged bit decoder and publish register-ready BF16 A fragments in smem.
- WG1/2: independent RS-WGMMA consumers, each owning64 output channels and sharing X. Native
  widths32/64/96/128/160/192/224/256 cover count classes64/128/192/256. The producer/decode WG
  receives88 registers/thread; both consumers receive208. Entry168 x384 and(88+208+208)x128
  both equal64,512 registers. All8 Hopper specializations compile without spills or serialization.
- Both consumers release the final X/A slot after WGMMA wait0, before their independent epilogues.
  Production/decoding of the next tile can proceed while an epilogue runs. One consumer can also
  start its next tile while the other finishes its epilogue. This is paired channel ownership;
  it does not alternate complete expert tiles between consumers. W2 writes256 per-warp norm
  partials per row, summed by F4, removing the heavy CTA-wide norm reduction/barrier.

S2 keeps wait1: each consumer releases the prior stage only after the next stage's first WGMMA
group has retired it; the last stage releases after wait0. The empty barrier requires both
consumers. Raw packet completion precedes decode; decoded A and TMA X each have a full barrier.
The CTA grid is one per SM, striding over expert-major tiles, so most fixed-fixture M128 CTAs
have successive jobs and an opportunity for cross-tile epilogue overlap. No global work queue.

W13 first gathers current BF16 activations into expert order **inside the timed call** so TMA
can read contiguous X; W2 reads current ACT directly. Gather is explicitly charged9.02/8.96us
in the model. Each packed weight tile is read/decoded once at M<=256; two output consumers
halve activation rereads relative to one consumer. This does not eliminate all activation
rereads across output tiles: the model assumes their remaining cached service hides under
the weight stream. `prepare()` adds only a byte permutation of existing NVFP4 codes/scales,
with exactly the same byte count; [packet_check.py](packet_check.py) inverts it bit for bit.

Local hardware uses real cp.async/mma.sync stand-ins. The same decode, per-warp norms,
stream/capture glue and ownership barriers execute. Local MAX64 retainsS2; larger classes
useS1 to fit the local99KB smem limit. Hopper usesS2 everywhere, with116,736/149,504/182,272/
215,040 dynamic bytes plus1,024 static bytes. The largest fitsH200's227,328B opt-in budget.
Native TMA, dynamic register redistribution and the large two-slot execution remain unverified
until the H200 receipt. PTX contracts: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
(`setmaxnreg`, `cp.async.bulk`, `mbarrier`, `wgmma.wait_group`).

The rejected initial512-thread implementation hit C7602 at N256 (WGMMA operands exceeded the
128-register target). Replacing it with the384-thread/88+208+208 allocation resolved compilation
locally. The rejected implementation is not retained or queued.

## Closed recipes and the changed mechanism

`8ee003db` cluster sharing lost1.13-1.44: it changed activation transport/ownership without a
separate weight producer and decoder pipeline. This candidate has no multicast, inter-CTA
synchronization or cluster launch; it pays a measured gather and uses CTA-local TMA rings.

`9cd95e72` producer/consumer SS-WGMMA also lost (best1.1305). It decoded BF16 into smem for an
SS consumer and lacked activation register reuse across tiles. Here decoded A is laid out
for direct register fragments, two RS consumers share X, three decode warps run independently
of TMA, registers move to the consumers, and per-warp W2 partials permit release-before-epilogue.
The smem round trip and full/empty handshakes still cost instructions; specialization alone
does not establish that they are hidden. The paired lab tests this hypothesis.

`ea794250` shared queues lost1.034-1.107 and replaced <=8 NT1 withN8, growing activation traffic.
Here NT1<=8 and N16G4 for9-16 tokens are unchanged; there is no global queue or DAG readiness.

## Predictions, not H200 results

The50.5/67.0us old residual includes light work. Heavy specialization targets only29.04/47.00us;
the light21.50/19.98us remains. [model.py](model.py) derives [model.json](model.json) from the
round3 ledger. Unique heavy weights at4TB/s take28.31/49.55us. Add4us for fill/drain, terminal
work and the larger finalizer reduction; assume heavy decode, MMA and cached X hide under this.
Then add unchanged light130.03/114.35us, shell16.00/17.08us and gather9.02/8.96us. The point
estimate assigns **zero interbranch overlap**: the confirmed64,512-register entry footprint
prevents an NT1 CTA from co-residing on the same SM. Spatial/tail overlap may still happen;
retaining the old credit clamped to32.31/0us gives a separate optimistic scenario155.05/193.94us.
This revises the earlier155/194 point claim on resource evidence. The4us allowance remains a
hypothesis; launch/tail were not separately identified. Reset attribution tests that partition.

| Arm | M128 us | M256 us | Basis |
|---|---:|---:|---|
| S14 control |185.43|261.03|Observed8ee003db, context182.4/256.4 is an earlier sample|
| CTA2 control |177.84|248.75|Prior model, pending reset batch|
| WS_TILE_BARRIER |190.27|198.64|Same pipeline, both epilogues gate next tile; exposes10% heavy residual as sensitivity|
| WS_PIPE |187.36|193.94|Cross-tile streaming; heavy residual and cached X hidden|
| PIPE optimistic spatial overlap |155.05|193.94|Requires interbranch overlap despite absent same-SM co-residency|
| Target |132.60|158.50|Requested whole-MoE target|
| PIPE point still above target |54.76|35.44|Light residual, gather, shell and unmeasured tail remain|

The barrier-arm10% residual is an explicit unmeasured sensitivity, not a fitted estimate.
In the optimistic M128 scenario both arms fit inside the assumed overlap window, giving no
whole-MoE gate difference. The point estimate instead predicts2.90/4.70us saved by PIPE.
PIPE point ratios are1.0104/0.7430 vs observed S14 and1.0535/0.7796 vs modeled CTA2. This
predicts a substantial M256 gain, a possible M128 loss, and **does not meet either target**.
The direct controls measure whether actual spatial overlap rescues the M128 result.

## Screening and admission

After the root-disk-full interruption all current source/evidence files were checked for
truncation. No round4 commit or sender had landed. The final source was regenerated, compiled
and screened again; earlier queued round2/round3/reset snapshots were left unchanged.

- Both `tools/local_eval.py --tests-only` forced adapters pass all5 official tests and all5
  static checks including sm90a compilation. Final JSON includes the exact adapter SHA256.
  Worst official relativeL2/row0.003805/0.004169.
- [check.py](check.py):1,036 checks,128 captured input cases, sizes33/127/128/255/256/257,
  empty/hot/real routes and all class boundaries, NaN ACT/Y/SS/OUT/GX/HP, two replays, full finite
  heavy256-way norm coverage,12 separately captured local_start64 checks. WorstL2/row
  **0.004212/0.004782**. CTA2 equals S14 on128/128 cases; each new arm equals S14 on48/128
  because norm summation order changes. Barrier and PIPE equal each other on128/128 cases.
- [audit.py](audit.py): O3 sm90a compile plus cuobjdump, all8 specializations have zero stack,
  spills or LDL/STL, no C7512/C7513/C7519/C7520 warning, retained wait1, tensor-X `UTMALDG.2D`,
  packed-weight `UBLKCP.S.G`, and88→208 register redistribution. [resources.json](resources.json)
  records instruction counts, registers and smem. An additional CFG audit follows all branches
  reachable after producer deallocation: max physicalR49 (MAX256) orR41 (others), below88;
  no producer path reaches consumer allocation, and every physical register is below208.
  Compile/resource admission is not runtime proof.
- [protocol_check.py](protocol_check.py):442 ownership/progress checks including400 varied
  completion schedules acrossS1/S2 and gated/PIPE, three successive32-stage tiles, static tile
  coverage at grids1/2/132/264 and expert counts through1024, and all256 HP entries.
  Negative controls catch S1 lag-release deadlock and S2 early-release overwriting a pending
  WGMMA X read. All200 PIPE schedules exhibit cross-tile stream/epilogue overlap in the model.
- [packet.json](packet.json) verifies inverse byte equality and packet alignment for three shapes.
- [entry_check.py](entry_check.py) / [entry.json](entry.json): the exact weights=None import entry
  with a1GiB cold-read callback completes24 seeds (72 fixtures/M/arm) in57.9s of its150s local
  budget. All arms pass; worstL2/row0.004132/0.005018. Local PIPE/S14 ratios1.395/1.355 lose;
  these use mma.sync stand-ins and S1 for large classes, so they do not rank native WGMMA.
  Rehearsal includes current local JIT cache; native H200 compile time is still charged to150s.

Local timing is not H200 evidence. No winner is promoted before paired whole-layer verification.

Decision rule: send independently of reset outcome, since S14 and CTA2 are both direct controls.
Compare each new arm's whole-MoE paired ratios against both controls at each M; gated-vs-PIPE
isolates the cross-tile epilogue dependency. A win must survive many seeds, then whole-layer
integration. If both lose, do not promote; use the reset phase attribution and the gate ablation
to distinguish exposed staging/issue cost from unavailable epilogue overlap.

## Queued

Attention import host **muse_attention_decode**, packed sender **4ca41bdc7a25cce7**, PID **237268**,
priority2, total budget150s shared with the unchanged round3 payload first. This second payload
receives all time remaining after round3. Original8c4c89c1 was stopped only while unsent under
the dispatch lock; no duplicate judge job. Exact metadata: [queue.json](queue.json); progress
`runs/detached/4ca41bdc7a25cce7.log`, snapshot
`runs/sent/4ca41bdc7a25cce7/lab_tail_payload+ws_payload.py`. No judge job ID yet.

`review/packed_entry.json` rehearses the actual shared150s wrapper deadline:22/24 seeds for
round3/4, both pass,106.4s total locally. Native H200 compile time is not in that cached local
estimate. Stage3 `8792099636654bac` loses all topology arms; round2 is at the judge as
`dd539517bc8344dd`. No production promotion. Next: read the paired receipts and use the
actual control spread and seed count before any whole-layer integration nomination.
