# P5 single-launch sliding attention

Structural replacement for sliding attention in frozen S11 final (`acda6c62`). One launch consumes the
actual QKV projection (BF16 or strided FP32 partials), performs Q/K RMSNorm and RoPE, appends the new KV,
gathers every causal key in the 2,048-key window, reduces split outputs through cluster DSM, and applies
output RMSNorm and the optional sigmoid gate. S11's separate gate-split choice is retained by the layer lab.
There are no metadata, global-counter reset, or combine launches.

`build.py` reuses only the vq8 low-level native FP8 primitives and projection-input loaders; `kernel.cu.inc`
contains the new algorithm. Each CTA owns one request/KV head/GQA subgroup/window split. A hardware cluster
owns all splits for that output. Historical KV prefetch precedes the PDL dependency wait. The new tokens use
an N8 QK and K32 PV tile. Cached tiles use N64 or N128 QK, LDSM.T+PRMT V transpose and register P for RS WGMMA.
The DSM reduction distributes output rows over cluster CTAs, with a final cluster barrier protecting lifetime.

`driver.py` queries actual occupancy before capture and chooses a resident cluster from shape/architecture.
It uses neither sequence values nor input data for dispatch. Batch 1 exposes four GQA subgroups per KV head;
larger batches keep all eight GQA heads together. Native variants have 124–168 registers and no spills or
C75xx warnings in the sm90a compile. Generated CUDA SHA-256: `e43cb5134924fff5e3f246759922b557259291e830804892a67697ee27381ca1`.

Verification completed locally on GPU0:

- `check.py`: all attention TASK cases and all six scored sliding shapes, N64/N128, BF16 and three strided
  FP32 splits, fresh-input graph replay, page-size/scale/boundary cases: **48 checks, zero failures**.
- `smoke.py`: the exact embedded whole-S11 payload, all six sliding shapes, two seeds, real QKV projection,
  original attention cache/output gates, CUDA graph stage mapping and CUPTI summaries. Every shape completes.
- Local timing is not H200 evidence. The first lab wiring bug (variant functions bound to the wrong globals
  dictionary) was caught locally and fixed before sending; the timeline now observes the actual attention node.

H200 measurement: `build_lab.py` writes `prof/lab/cluster_sliding_timeline.py` (`6236a0f6`). Job
**7906118231f04058** on the prefill host landed: **target missed**. It fixes S11's selected projection/expert/prefetch plans,
compares A/A2/N64/N128, uses four seeds × eight rotating cold-L2 replays, and reports original graph
CUPTI completion increments. `CS_XZ85` is a lossless compressed JSON archive of paired seed times, resource
plans, output drift, gates and attention node durations. No structural kernel is promoted without this result
and subsequent whole-layer correctness/promotion checks.

Integration's already-measured decode-1 direct-partials win is separate:
[handoff](../vq8/HANDOFF.md), commit `d4d2e76`, patch `d454609d`, onto S11 `acda6c62`.

| Sliding shape | S11 increment us | N64 increment / layer ratio | N128 increment / layer ratio | A2 ratio |
|---|---:|---:|---:|---:|
| decode 1 | 14.752 | 14.912 / 1.01061 | 14.176 / .99582 | 1.00811 |
| decode 16 | 15.680 | 23.456 / 1.08216 | 20.640 / 1.04966 | .99316 |
| decode 32 | 18.273 | 38.048 / 1.09986 | 23.904 / 1.02058 | .99888 |
| verify 1 | 17.376 | 12.768 / .96929 | 12.640 / .97023 | .98937 |
| verify 16 | 24.768 | 24.640 / 1.00123 | 25.536 / 1.00441 | .99755 |
| verify 32 | 28.832 | 27.904 / 1.00260 | 29.697 / 1.01002 | .99775 |

All native attention gates pass. The host passes official prefill correctness (1.43741x host, unrelated to
variant promotion). Whole-layer output/routing drift is recorded in `h200_7906118231f04058.json`; this is
not an official whole-layer TASK pass for the variant. Auto cluster sizes are genuinely 16 at batch 1,
9–12 at batch 16, 5–7 at batch 32, so the loss is not a one-CTA launch fallback. The verify-1 time gain
needs stronger paired/correctness evidence before any integration handoff.

Follow-up: `build_warp.py` uses 32-thread CTAs and m16 FP8 MMA at decode, preloads page IDs together and
skips page loads for nonexistent pipeline tiles. `build_warp_lab.py` compares auto/fixed4/fixed8 split plans
in S11, with phase clocks in separate instrumented native and warp kernels after the actual QKV producer.
Local testing caught an output-row loop that still advanced by four warps after changing to one warp;
fixing its stride makes all 24 decode TASK/FP32-partial checks pass.

H200 follow-up **c0fd644c2bef4a7a**, source `2b4b170f`, also loses (four seeds × eight replays):

| Decode sliding | S11 increment us | Warp N64 increment / ratio | Warp N128 increment / ratio | Fixed4 ratio | Fixed8 ratio |
|---|---:|---:|---:|---:|---:|
| B1 | 15.360 | 14.720 / 1.01306 | 17.665 / 1.06165 | 1.18296 | 1.10417 |
| B16 | 14.688 | 18.208 / 1.02605 | 17.793 / 1.05472 | 1.09725 | 1.04087 |
| B32 | 18.560 | 32.192 / 1.09066 | 43.104 / 1.13203 | 1.05577 | 1.11110 |

A2 controls .99951/.99948/.99879. All attention gates pass; no promotion. Instrumented B1 native/warp
preparation after the dependency wait takes 5.28/5.12 us median; gather/virtual tile 1.79/2.75 us. Native
critical max stamps: entry spread 4.736, projection-ready 5.408, prepared 11.264, gather complete 13.120,
partial barrier 13.888, reduction complete 15.424, final barrier 15.872 us. The warp design's lower math
waste does not overcome its issue/reduction costs. Higher-batch native clock arms used the warp arm's
cluster plan and can have extra residency waves; their entry spreads are explicitly retained in the receipt.

Next structural fix is cooperative projection reduction (`build_compact.py`): one warp per **real** Q/K/V
row, one load pass over all projection partials, and metadata/RoPE/zeroing before the dependency wait.
The prior design redundantly loaded padded Q rows and serialized Q and new-KV input passes.

Cooperative-prep H200 job **915040fb98e841b5** (`2ae16f19`) wins decode 1, with four paired seeds ×
eight cold replays. It still misses the requested latency target.

| Decode sliding | S11 increment us | N64 increment / layer ratio | N128 increment / layer ratio | Fixed4 ratio | Fixed8 ratio | A2 |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 14.912 | 10.912 / .96487 | 12.288 / .96387 | 1.01094 | .96913 | 1.00428 |
| B16 | 14.592 | 14.720 / 1.01770 | 12.192 / .99221 | .99816 | .98773 | .99735 |
| B32 | 18.208 | 26.368 / 1.02335 | 18.176 / .97677 | .97915 | 1.03759 | .99304 |

All attention gates pass. B1 native/prep-replacement median input/norm/rope/store phase falls from
5.28 to **1.824 us**, and clocked gather/virtual tile 1.792 to 1.472 us. Its component graph (actual QKV
producer + attention) ends 11.264 us from first CTA entry, projection-ready at 5.024 us. The actual whole
layer attention increment is larger, consistent with downstream PDL kernels competing for resources;
that cause remains a hypothesis until the next paired policy study. B16/32 whole-layer changes include
allowed numerical routing changes, so the small ratios do not establish an attention-path promotion.

`patch_cluster.py` builds an experimental S11 decode-1 candidate (`842eb8eb`, N128 auto cluster).
Its **normal and forced local_eval audits both complete**: 13 TASK tests, 12 fresh graph shapes, 24 static
checks each. An initial report-write failure was fixed by creating the output directories and rerunning
both audits; the successful JSON is under ignored `out/{normal,forced}`. This is not yet an integration
promotion: compare against the already-handed-off direct-partials winner over many seeds first.

Next: `build_gated.py` stages the gate in the same cooperative load pass (in unused KN storage), then
reads it from shared memory in the epilogue. `build_gated_lab.py` compares S11 gate split, one full-QKV
projection + one gated attention launch, and PDL dependent release after preparation (eight paired seeds).
The shared-gate kernel passes all 24 decode TASK/FP32-partial checks, and every policy runs inside the exact
local whole-layer/CUPTI lab. Payload `ea59b980` landed as c773699a977f4574; results below.


Shared-gate/policy job **c773699a977f4574** (eight seeds × eight): C128 gate-split whole-layer ratios
.99005/.99011/.96795 at decode B1/B16/B32. Full-QKV fused N128 gives 1.00258/1.00492/.96837;
late dependent release gives 1.00191/1.00331/.96890. Fused N64 gives 1.00063/1.01767/1.00193.
The B1 increment drops 14.95→9.70 us with full-QKV fusion, but its larger producer offsets that saving.
Retain the original split-gate producer. A2 .99855/.99852/.99551; no new policy promoted.

## Nonspinning last-CTA reduction

`build_last.py` replaces cluster DSM with global FP32 partials, a release/acquire arrival counter, and a
nonspinning last-CTA epilogue. It can distribute splits without cluster residency constraints. No reset or
combine kernel exists; captured attention consists of exactly one launch. `last_driver.py` uses shape-only
split selection and caches zero-initialized counters before capture. Counter keys exclude capture-stream
identity, so graph capture cannot insert a hidden zeros launch. Native N64/N128 use 98/138–139 registers,
32 stack bytes from the RoPE helper, zero spills and no C75xx warnings. CUDA `3815beab`.

**7e93cc81af17489a**, decode host, combined structural lab, 16 seeds × eight cold replays:

| Decode sliding | S11 increment us | N64 ratio / increment us | N128 ratio / increment us | Fixed4 ratio | Fixed8 ratio | Direct-partials | A2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 15.10 | .97994 / 13.76 | **.95868 / 11.10** | 1.01766 | .97818 | .97811 | .99221 |
| B16 | 14.53 | 1.01092 / 14.59 | .98881 / 14.08 | 1.01514 | .99094 | .99371 | .99615 |
| B32 | 18.18 | 1.01542 / 20.29 | .99545 / 18.24 | .99833 | 1.02796 | .99648 | 1.00406 |

B1 C128/direct-partials .98017, SE .00131, saving 1.43225 us; C128/S11 saves 2.98375 us. Handoff only
B1 sliding: [patch and validation](HANDOFF_LAST.md). Whole-layer critical-path target remains missed.
All native gates pass and full replay outputs are finite; native changed-candidate official check is pending.
The combined check's unchanged S11 host passes 13/13, 1.71559x, CI [1.68986,1.74133]. Receipts and paired
seed archive: `../full128/h200_7e93cc81af17489a.json`, `runs/muse_decode_layer/20261025T100820Z_check_7e93cc81af17489a.*`.

Independent `check_last.py`: 24 fixtures × eight fresh graph replays, zero counter residues. Bare
`fead4a00` candidate normal and forced: 13 TASK, 12 graph shapes, 24 static checks each. The exact next
wrapped check also validates the register-only full128 follow-up in prepare; heavy local audits run serially
after simultaneous audits exhausted GPU0 memory. Failed OOM screens stayed local and successful screens reran cleanly.
