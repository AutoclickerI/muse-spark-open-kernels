# P5b full attention and S15 audits

Full decode16/32 and verify16/32 remain active. Exact S15 base is `60254883`.
P5 owns sliding/decode1; this lane does not edit those implementations.

| Reset batch | Job | Native result / decision |
|---|---|---|
| Verify softmax, S14 N64 | `457f3c73b3214534` | All gates pass; mix25 1.05012/1.04610, tc25 1.03473/1.03452, ping 1.15929/1.17091. A:A .99941/.99947. No winner. |
| Decode softmax, S14 attn_main | `e4fcf732f35b425d` | All gates pass; mix25 1.00182/1.00284, tc 1.00976/1.00704, padded-WG reference 1.33968/1.28368. A:A .99964/1.00105. No winner. |
| S15e PDL release | `f88fda4aac144374` | Repaired entry-only wrapper accepted; complete; selected edits enter held-out confirmation. |
| S15L library boundaries | `f22020d25d234be6` | Repaired entry-only wrapper accepted; complete; selected edits enter held-out confirmation. |
| S15/S15n/SGLang,65 shapes | restored sender587925 | Complete a793c99eb9d447df; nearby/scored ratios improve strongly, guarded native TASK follow-up sent. |

Ratios are new/old time; lower is faster. Per-recipe numbers, A:A ranges and
phase clocks are in `softmax/reset_result.json`, `decode/reset_result.json`
and FINDINGS Attention. The losing recipes do not close alternate exp layouts,
P*ones without a separate wait, native K-major q1 WGMMA, or operand-ready rings.
No P4 promotion from either stage batch.

## Structural follow-up

`pvsum_build.py` moves the N128 denominator into four extra PV accumulator
registers. It issues P*ones in the existing PV WGMMA group, preserving native
MUFU and online max/rescale. Independent packed-FP8/rescale probes have zero
error over three seeds; sm90a compilation succeeds without C75 warnings.
The main kernel adds16B/thread spill stores+loads at168 registers; that cost is
explicitly measured rather than described as a no-spill win. The unchanged
combine already spills. `pvsum/probe_native.json` records both.

`gather_build.py` changes only full q1 B16/32 page/row issue: consecutive quads
per loader and vectorized page-ID loads where four rows are valid. Page/tail
handling preserves every key. The layer adapter now satisfies the driver's
raw-qkv metadata ABI while preserving the existing PDL BF16 sum/cast boundary.

`build_next.py` packs A,A2,V,V2,G at all12 scored shapes, eight fresh seeds and
eight rotated cold graph replays. The native denominator/rescale probe precedes
timing, every arm/seed uses the unchanged TASK checker, and per-seed absolute
shape times are retained for the six-load score confidence interval. V2 is an
identical executable control for V. No arm is promoted before native TASK and
control gates and a held-out six-load score confirmation.

The exact host wrapper exposes a top-level kernel_fn for judge AST admission.
Local screens substitute only Hopper launches; they do not prove native
WGMMA concurrency. All remote experiments go through h200_lab.

## Nearby correctness follow-up

`../` is attention; the layer audit lives in `muse_decode_layer/dev/nearby`.
The original S15n range admits full verify above B64 despite NSMAX64. The
minimal guarded patch limits that specialization to B<=64; none of65 timed
cases changes. The67-shape native TASK payload (adds B64/B65) passes268 checks
per local mode and is ready for a positive range-dispatch timing outcome.

Repaired audit senders are P4's exact entry-only fixes. The three unsent originals
were retired under the dispatch lock only after identity checks; each audit's
`reset_superseded.json` preserves its original metadata and repaired identity.

Structural follow-up queued: **sender589881**, exact wrapper **18895a35**, decode check, priority6, past not-before1792972800. `next/queue.json` and `next/admission.json` record identity:480 TASK gates,13 official host tests and2 static checks pass in each mode. The automatically merged unsent sender579268 was retired before acceptance so the65-shape nearby audit and this payload keep their full independent prepare deadlines; `next/auto_merge_retired.json` records that event. Nearby was restored byte-for-byte as sender587925 at priority4.9.

PDL/library follow-up: sender760641, wrapper7127261f, fifteen arms and1,440 TASK gates per local mode; all143 C fixtures and25 static checks per mode pass. Exact group patches and receipt controls are indexed by `muse_decode_layer/dev/pdl_release/P4_HANDOFF.json`. This is a nomination pending held-out six-load R. Structural attention job is fb14f30ffd48450f; nearby job a793c99eb9d447df.
