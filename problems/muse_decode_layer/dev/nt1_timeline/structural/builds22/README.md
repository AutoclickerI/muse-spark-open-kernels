# S22 builds for the operand diagnostic outcomes

**Measured, receipt `f6970526aa6f4a28`: every arm loses whole-load six-load R vs S22 `4b8c803d`.**
CACHE1 0.9613 [0.9601,0.9624], LAZY 0.9046 [0.9030,0.9062], LAZY2 0.9039 [0.9023,0.9056],
N1SIMT 0.8118 [0.8082,0.8154], EPI4 0.9983 [0.9979,0.9988]; A:A 0.9987 [0.9982,0.9991]; 5376 TASK
+ 780 nearby pass. Per load and per shape: [REPORT.md](REPORT.md), `results.json`. The reader exports
EPI4's d1S whole-layer point (time 0.9912 [0.9885,0.9940]) to `P4_HANDOFF.json`, but it does not
qualify: the identical-S22 twin A2 is itself 0.9946 at d1S (log SE .0014), so EPI4 vs A2 is about
0.9966 (CI spans 1) and its d1S-only six-load estimate is ~1.0004 with a CI spanning 1; EPI4 loses
at d1F (1.0043) and every M>=8 shape. No P3a arm goes to P4 from this receipt.

| Arm | Outcome it targets | Correct implementation |
|---|---|---|
| CACHE1 | Shared/register fragment placement and lifetime | Fill a real64KiB two-plane BF16 bank;one raw TMA slot keeps two CTAs |
| LAZY | Fragment register lifetime | Original BF16 loads per K-group/tile |
| LAZY2 | Accumulator recurrence | LAZY plus two independent FP32 chains and final sum |
| N1SIMT | Useful-N MMA padding | Correct FP32 matvec for actual singleton token blocks,HMMA otherwise |
| EPI4 | Epilogue serialization | Four warps with scalar W2 channel ownership;one norm flush |

The old shared-fragment diagnostic bank is a synthetic oracle. CACHE1 fills
the full real bank inside the timed kernel and pays publication/LDS/one-slot
costs. N1SIMT retains every K element and expert;it does not ship the diagnostic
PAD_USEFUL shortcut. LAZY2 uses register capacity freed by its own operand
lifetime change;the old cached-Bf ILP result does not measure this recipe.

`base_snapshot.json` pins the original S22 Git blob. `build.py` reconstructs
the base and exact candidate/shipped files in ignored `runs/screen/p3a_builds22`.
Five checked-in patches apply to **expanded** `dev/s22/generated/s22.py`;
`manifest.json` pins their hashes. The original third-party notices remain.

`leaf_validation.json`, `screen_validation.json`, `counts.json` and
`rehearsal_validation.json` contain the local evidence. Local CACHE1 substitutes
only its over-limit shared activation-bank storage with preallocated global
storage;native H200 bank transport is unverified. Other four MoE math bodies
run natively on sm120;other Hopper stages use S22's existing forced stand-ins.
The exact packed local rehearsal uses two independent local seeds/two rounds;
the transported native plan is64disjoint seeds×16symmetric rounds.

`lab_manifest.json` declares all12whole graphs for every arm,exact S22 A:A,
TASK/nearby counts and ec754e5 all-round protocol. `sample_codec.py` losslessly
packs existing six-decimal primary samples and deterministic round orders.
Four worst full traces retain GPU/enqueue times and NVML. The synthetic packet
capacity fixture is explicitly not a performance measurement.

`follow.py` is a finite receipt consumer,not a model session. It waits on actual
file/sender events,uses the predeclared upper-bound rules in `refuted`,and can
replace only this owned still-local waiter under the dispatch lock. Once the
judge accepts the batch,all arms remain. A failed diagnostic audit closes
nothing. The one eventual paired receipt is audited by `read.py`,then exact
P4 nominations appear in `P4_HANDOFF.json` and parent `point_nominations.json`.
The worker neither scores nor edits shared docs. `follow_seal.json` pins code
and initial queue identity so changed work cannot be silently acted on.

G0 uses six-load whole-layer R,not a transfer of per-call gains. Every valid
per-shape point win nominates;P4 composes and confirms on new held-out seeds.
