# S22 builds for the operand diagnostic outcomes

[Current report, budgets, screens and receipt state](REPORT.md). P3a is active.
The diagnostic remains sender3637718/wrappere1710fb3. The single production
batch is sender984504/wrapperab36c11e7cb7ea6c,queued on the MoE host at rank4
after09:00KST10-27. No new H200 result exists yet.

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
