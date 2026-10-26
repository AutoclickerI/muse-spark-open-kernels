# P5b review: three structural targets, one paired payload

Exact base S15 `60254883`. `claims.json` checks each premise against extracted
S15 CUDA, with line numbers. `build_rebuild_lab.py` packs all arms against the
same prepared S15 state:12 shapes, eight fresh seeds, eight rotated cold replays,
A/A2 and854 public TASK checks (all13 official cases per arm, eager/replay, plus the12 scored shapes over eight seeds). Per-seed shape times retain six-load covariance.
Native job1bab0b2d0dc14931 passed every gate; measured results and nominations are below.

| Arm | Confirmed premise and change |
|---|---|
| ready | Producer's shared full[slot] arrives after V transpose; QK(next) waits it. Publish a separate K-ready after cp.wait/proxy fence; QK waits K, every PV explicitly waits V. Adds32B static shared at ST4. |
| ready_elide | Starting with ready, remove only PV(current)'s wait inside step: all four K(next) arrivals imply all four earlier V(current) arrivals. Retain final PV's explicit V wait because there is no K(next) dependency. |
| kmajor | S15 q1 widens FP8 K into FP16 fragments for warp MMA, padding eight query rows. Use native FP8 register-A WGMMA M64keys/N8queries for cached QK. Existing gather64, split plan, FP16-P PV, new-token path and output merge remain. |
| reg | Original single-state VQ8 math, producer32/consumer232 setmaxnreg only. Controls the register-transfer overhead. Initial168-register launch allocation stays within64K; transfer budget63488. |
| parity | Two statically indexed online max/sum/O chains for even/odd cached tiles; keep QK(next)/PV(current) overlap, include new-token tile, merge both states in the log domain at the portion end. Same register transfer as reg. |

The three original claims are present; none is dropped. Native SASS confirms
QGMMA.64x8x32.F32.E4M3.E4M3 for K-major QK, rather than the emulated mma.sync
FP8 path. It also confirms register alloc/dealloc instructions. All final
sources compile at O3 without C75 pipeline-serialization warnings. Resource
costs are evidence, not admission claims of speed: K-major full q1 uses116
registers and no spill stores/loads; ready/ready_elide have8/4B spills and
reg/parity have20/32B spills at the stage4 specialization. Final native.json
records all stages and unchanged combine resources. SASS reaches R193/R200 in
reg/parity while producer execution stays below R32, proving the consumer code
uses registers above the168-register entry allocation; neither exceeds232.

GPU0 normal/forced screens run in the judge image. The original local VQ8 N64 stand-in
changes the N128 reduction order and fails the same v16/full routing gate in A, A2
and every changed arm (token103, .0315 margin versus .03). All854 gates remain;
`n64_admission_failure.json` preserves both failures. The original local N128
fallback also lacks `&3` on WG1 Q indexing; the repaired local-only stand-in
fixes that index, preserves N128 arithmetic and stages exact transposed FP8 V
bytes in global scratch to fit sm120 shared capacity. Both repaired normal/forced screens now pass854/854TASK checks and13official
host/2static checks:173.85/124.29s private lab. N128 alone retained the routing
miss; preserving the original VDSK qkv tile/split and reducing only its prefetch
depth8→3 prevented the172KiB resource rejection and generic projection, resolving
the miss. The prior N64 and N128-only failures remain in their JSON artifacts. Native setmaxnreg/WGMMA are compiled separately.
K-major's portable helper uses exact E4M3 operands and the same score mapping.
An independent native small-integer K/Q layout probe must pass before timing.
Every actual whole-layer arm is gated by TASK.md, including routing/cache and
fresh-input graph replay. No loss is inferred from a failed control or execution.

Send one decode-host import-phase check (620s private lab budget; exact S15 real
host prepare unchanged). Private synthetic weights/state are discarded after
the lab; no result feeds timed returns. No prefill host is used. Shape-point
improvements go to P4 regardless of A:A clearance, with uncertainty; P4's held-out
six-load R lower confidence bound is the final promotion gate.

The first local packed output exceeded its8400-byte archive limit (9475/9715B); it stayed local. Final recording retains every arm, seed, gate and diagnostic, with seed times at .001us precision. K-major converts normalized Q once per CTA and reuses it across cached tiles. Slim pointer-only C++ bindings reduce cold compile overhead without changing any device code.

Orchestrator explicitly authorized the native diagnostic at11:58KST despite the
shared local control failure, with that seed and every TASK gate unchanged.
Sender1507783 (priority3, decode check) is job `1bab0b2d0dc14931`, exact wrapper
`16f996bc`, payload `7b2a69b9`. Native correctness remains a gate, not an inference
from S15 passing earlier judge seeds. Every verified point winner is nominated;
P4 confirms held-out paired six-load R lower95% bound above1.

## Native structural result

Job **1bab0b2d0dc14931**,854/854TASK checks,8seeds×8cold replays,262.9s
private lab. All3native K-major layout probes return exactly0error. This proves
the retained routing seed passes real S15 and every native arm.

| Arm | affected full shape time/S15 (SE),16/32 | Decision |
|---|---|---|
|ready|1.001874(0.000562)/1.007053(0.000688)|tested form loses|
|ready_elide|0.997130(0.000760)/0.998618(0.000929)|nominate both points|
|reg|1.015525(0.000933)/1.033249(0.001325)|tested form loses|
|parity|1.033005(0.001895)/1.048299(0.001689)|tested form loses|
|kmajor|1.193606(0.006464)/1.243381(0.003908)|tested form loses|

Affected-shape A:A time ratios v16F/v32F=.998796/.999073;d16F/d32F=1.002203/.999249.
Full six-load A:A R=.996636,95%CI[.994997,.998278] shows capture drift, dominated
by unchanged d1; retain it as uncertainty. No standalone arm clears G0.
Ready_elide R=.996507[.994361,.998658] is not a global promotion: its source-wide
aggregate includes unchanged cells. `P4_HANDOFF.json` nominates only the two
changed verify-full cells with8seed paired shape SE and exact patch.

Parity/reg=1.017210/1.014567 (SE.001356/.001194): the second state adds cost
beyond register transfer. This closes the tested parity schedule, not all
independent-state overlap. The current K-major form synchronously writes and
reads a shared score matrix between native QK and the old warp softmax/PV; its
19.4/24.3% whole-layer losses leave a direct accumulator-to-softmax mapping open.
The explicit V-wait split loses while the proven redundant-wait elision wins
slightly; producer V-transpose overlap remains a structural lever.

## K-major occupancy response

`kmajor_reuse.patch` reuses the consumed K ring tile for score remapping after
both head QK operations and an explicit CTA barrier. Gather, split plan,
probabilities, PV and native instructions remain unchanged. Native O3 now
uses114registers,zero spills and2048Bstatic shared (old116/zero/6144).
Together with73984Bdynamic this removes the shared-memory bound from three
resident CTAs on H200: old3×80128=240384>233472B; new3×76032=228096B.
The next lab records actual native occupancy, plus S15/A:A/old K-major/reuse
at12shapes,8freshseeds and488public TASK gates. This is a tested-cause response,
not a claim that the original K-major design was fast. Local screens pending.
