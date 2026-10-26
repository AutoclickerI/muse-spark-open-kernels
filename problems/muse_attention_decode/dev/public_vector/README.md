# Production attention transfer and vectorized direct partials

Control: exact S21 shipment `eacc087b` plus accepted patch `9b043487`,
combined source `fbaf2793`. The experiment measures **whole `kernel_fn`
CUDA graphs**, with all projections, prep, gate and output kernels inside
the boundary. Each single-cell leave-out changes only that cell; the load
is `46 sliding + 16 full`. No attention-call ratio is a promotion input.

## What the existing production receipt establishes

Receipt **5864563117fd44b6**, 64 held-out seeds, unchanged TASK checker.
The following are **port/S19 whole-load time**, smaller is faster. The
grouped public leave-out isolates exactly one port in these three loads;
it does not isolate v1S from v1F or v16S from v16F.

| Cell | Port/S19 load time, 95% CI | A:A load time, 95% CI | Applicability |
|---|---|---|---|
| d16S | 1.043149 [1.042282, 1.044018] | .999769 [.999095, 1.000444] | Exact current d16 load; restore S19 |
| d32S | .994855 [.993713, .995997] | 1.000041 [.999549, 1.000533] | Before accepted direct-input change; remeasure |
| v32S | .996492 [.994566, .998422] | .999377 [.998992, .999762] | Exact current v32 load; control drift disclosed |
| v1S / v1F | unisolated | .999495 [.998753, 1.000237] | Grouped no_PUBLIC/S21 1.023489; two cells changed |
| v16S / v16F | unisolated | 1.000648 [1.000039, 1.001258] | Grouped no_PUBLIC/S21 .995999; two cells changed |

The grouped six-load no_PUBLIC/S21 is .998709 [.997343, 1.000077]. This
does not establish a net port gain. Those old call wins remain measurements
of isolated calls, not evidence for new layer nominations. `audit.json`
contains every source-backed boundary and the reconstructed prior CIs.

## PDL inventory and mechanism

Every current public main has no explicit wait/release and uses an
ordinary launch. Completion ordering makes it correct but prevents the
independent consumer prologue from starting during its producer's tail.
The FP32 pack is also an ordinary launch. These are source facts;
attributing a number of microseconds to them still needs the new receipt.

| Cell | Current timed attention kernels | Current release / wait | Successor and missing opportunity |
|---|---|---|---|
| d16S | pack, Lee attnd | exit / ordinary completion | PDL gate-split GEMM can compute independently until its final Y read; ordinary attention prevents that overlap |
| d32S | direct Lee attnd | exit / ordinary completion | NF O releases at entry and waits before attention reads; lost weight/prologue overlap |
| v1S | direct OP fused | exit / ordinary completion | Same NF O edge; no remaining pack in this accepted cell |
| v1F | pack, Lee attnp68 | exit / ordinary completion | Pack and main break the PDL chain to NF O |
| v16S | pack, Lee attnp32836 | exit / ordinary completion | Native QKV producer permits independent history gather; O remains an ordinary library boundary |
| v16F | pack, Lee attnp324 | exit / ordinary completion | Same incoming chain/packing boundary; O remains ordinary |
| v32S | Lee attnp32836, BF16 view | exit / ordinary completion | No pack to delete; upstream Lt and downstream O have ordinary library boundaries, limiting PDL-only leverage |

The owned NF prologue/wait follows `_nf_pdl_warp` / `_nf_pdl_wide` in
the exact S21 source; the gate reads Y only after `gdc_wait`. The v16/v32
library algorithms are prepare-selected: the new native trace retains
their actual adjacent names/timestamps rather than inventing an owned wait.

## Independent arms

At all seven ported cells, A/A2 are the current production composition;
S restores the exact S19 attention dispatch inside the otherwise identical
S21 graph. D enables PDL on the existing pack/main chain. Lee releases at
entry, initializes CTA metadata/barriers, then the history producer runs
without waiting for QKV. Consumers wait before any QKV load. EQ preload
moves below that CTA barrier, so it cannot stall the producer. OP releases
at entry and waits after its scalar independent metadata prologue.

V consumes projection partials directly with plane-outer 16-byte loads,
ordered FP32 addition and the existing BF16 rounding boundary. Lee LN0
loads each query and own K/V slice once per CTA; LN1 reuses the query tile
and BF16 own-KV staging in its unused query space **only when both lanes
belong to the same request**. At balanced request boundaries each lane
reads its own distinct rows; an idle peer never participates in barrier13.
There is no new shared
allocation. Consumer-only barrier 13 does not block the history producer.
OP loads contiguous 64-element rows, then splits even/odd RoPE lanes in
registers. F combines D and V. V has ordinary launches as its control;
the consumer preload moves in all new Lee arms, so D isolates PDL plumbing
including its necessary safe wait/preload scheduling.

Native d1S and d1F get separate V arms: 16 even lanes issue aligned float4
plane loads once, reduce in registers, and pass the other BF16 pair to the
odd lane. Gate vectors use contiguous float4 loads within each thread.
d1S edits the **actual `_s16_global_C128_cs_CUDA` last-CTA source**, not the
older unused cluster source. Its launch geometry and dispatch are unchanged.

The native Lee SASS contains LDG.E.128. PDL arms contain PREEXIT at entry
and ACQBULK at the consumer wait; V omits the PDL instructions. A two-kernel
assembly probe verifies those instruction meanings. Decode
attnd has no spills. Verify attnp still has some compiler spills; their
cost and two-lane shared-memory/barrier behavior remain native TASK/timing
questions, not a mechanism claimed solved by local emulation.

## Screens and queue

- `tools/local_eval.py` production-graph batch: **108 TASK checks each**
  normal and forced, 12 cells, two fresh seeds, eager/warm/capture/replay.
  Local timing fixtures use context257; no local timing ranks an arm.
- Exact ready candidate: **28 eager + graph cases each** normal and forced,
  including TASK attention cases and every scored producer ABI.
- Five exact sm90a CUDA units compile. The changed readers pass eight
  bit-exact tests: BF16 and ordered FP32 planes at S1/3/7/10. Local Lee
  arithmetic uses the explicit one-lane WGMMA stand-in because GPU0 cannot
  hold Hopper's two-lane shared ring. Native two-lane attention TASK remains pending. The exact two-lane balanced
  map/query-sharing prologue runs locally:15 cases,99 cross-request CTAs and
  one active/idle peer, every emitted query bit-exact and no stranded barrier.
- Full-size synthetic transport rehearsal: 7,756 bytes before native
  profiling diagnostics, exact timestamp round trip, 11,700-byte reservation.
- **Sender3831549**, wrapper`5191ed5db2ed9eb8`, attention host, check,
  priority2, **2026-10-27 09:00 KST / epoch1793059200**. All hosts had zero
  quota. Payload`b4a5ec1bb85d260e`: eight fixed fresh seeds
  `281730011 + 104729*i`, eight forward/reverse cold rounds, **432 TASK**,
  every integer-nanosecond round retained. Profiles capture A/S/F attention
  and adjacent graph nodes. `queue.json` matches the waiting sender's exact
  wrapper hash; this is one batched check, not a scored candidate send.

The exact candidate screen retains all 37 inherited native compile checks (43 static
checks total); the local cache adapter expands the packed source only for compilation
so its unit keys agree with local_eval. No checker or compiler gate was removed.

The all-F review patch is `6a605fa2`, source`a4c63f0b`, onto exact
`fbaf2793`. It retains complete MIT/Apache notices already in that base.
It is **ready but unmeasured**, not a new nominee. P4 already owns the
d16S restore; its exact one-line patch and prior CI are in `P4_HANDOFF.json`.

## Receipt decision

Read the job assigned to sender3831549, then run `analyze.py <receipt>`.
Require exact source/seeds/round counts, native completion and every TASK
gate. Report A:A with each interval; use the larger paired/A:A log SE,
never subtract control drift, discard rounds or retime selected seeds.
For each cell compare whole-load times of S/D/V/F, with A2 for its unchanged
counterpart. Hand correct improving points to P4 as per-cell patches; P4
confirms the selected composition on new held-out seeds under GOAL G0.
The simultaneous F six-load estimate is also archived. A failed native
implementation or control is unmeasured, not a closed structural recipe.

Rebuild: `public_hybrid/build.py` (if its ignored reviewed-source export is
absent), `build.py`, `build_lab.py`, `build_candidate.py`. All generated
binaries/logs/expanded sources stay in ignored `out/`. Only the reviewed
kernel bodies and full license notices are embedded; no donor scripts run.

## Overnight S22 follow-through

The outcome-specific builds are now in [night/README.md](night/README.md):
six screened production candidates on P4 S22 expanded4b8c803d, sealed worker
691620 behind diagnostic sender3831549. It sends one ec754e5 paired check after
reading/pruning the diagnostic. The original diagnostic and its S21+9b control
stay unchanged. The ready patches remain unmeasured until the native receipt.
