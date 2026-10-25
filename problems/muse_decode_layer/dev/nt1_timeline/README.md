# P3a small-M MoE: current work

Base S14 final `f1c77ce6`; flags `{1:95,8:69,16:69,32:69}`. No new kernel promoted.
The reset batch is unchanged. Its result is not required to send the independent second batch.

## Reset queue (unchanged)

- MoE sender PID3360739, wrapper `917e79f37f28b5c1`, payload `prof/lab/moe_nt1_epilogue.py` `2b0356d7`.
- Not before **1792972800 =2026-10-26 09:00 KST**; priority1.0; no judge ID yet.
- `runs/waiting/3360739.json`, log `runs/detached/917e79f37f28b5c1.log`.
- EPBASE/VEC/PROD/NORM/PAIR plus fixed-math ONCE contrasts;16seeds, exactS14 control.

## Second-round batch (screened and queued)

`build_epi_next.py` derives four cuts from the queued two-writer PAIR cell:

| Arm | Cell | SHA256[:8] | New work |
|---|---|---|---|
| TREE | `epiu_cell.py` | `30e50fdd` | Eight-way FP32 sum uses a balanced tree: native marked FADD chain8->3 |
| FOUR | `epif_cell.py` | `780b3cee` | Four16-lane writers instead of two32-lane writers, same modeled32 reduction bank wavefronts |
| LOCAL | `epil_cell.py` | `c9ce6f15` | Shared per-CTA norm sums, transpose export, parallel final gather; zero global norm atomics |
| FOUR_LOCAL | `epix_cell.py` | `b669c375` | Four writers plus disjoint local half-norm slots and final gather |

All retain Ksplit8, two packed stages and original dequant/MMA. No widened prepare weights.
LOCAL trades atomics for2KiB shared atM32,264KiB of current-call global export at grid264 and a finalizer gather.
`ISSUE.md` gives the address/dependency model and its limits. `issue_audit.py` compiles marked actual
bodies sm90a; `issue_budget.py` normalizes per16weights/lane against4TB/s; `reduction_depth.py` checks
actual add dependencies. `audit_epi_next.py` confirms every full native flags69 image has128HMUL2/32HMMA;
LOCAL variants contain zero FP32 global atomic sites. Native performance regs125/125/125/127, zero spills.

`build_epi_next_lab.py` -> `prof/lab/moe_nt1_epi_next.py`: all four arms with S14 and PAIR controls,
32weight/router seeds, M8/16/32, readflush AB/BAx6, direct PAIR ratios, A:A, CUPTI nodes/unique-byte rates,
16phase seeds at each M, observer tax/effectiveGHz, local export/final gather clocks and route issue model.
Synthetic weights; independent of the first reset outcome. All arms fit in one second-round run.

Queued on MoE: sender **PID1005738**, wrapper **e23854aca7a11da1**, payload **642d6ae0**,
priority**1.1** behind the first batch's1.0, not before**1792972800** (same09:00 KST reset).
`runs/waiting/1005738.json`, log `runs/detached/e23854aca7a11da1.log`. No judge ID yet.
The original sender3360739 and wrapper917e79f3 were verified unchanged after this send.


Each arm passed528 component/replay checks +6forced ABI launches in the judge image (`check_epi_next.py`),
including real/empty/one/all8 routes, zero inputs,8/16/24/32-token edges, three local-start offsets and fresh
captured inputs. Worst TASK L2 .0024375; max relative versus INC .000251. Full normal/forced screens run
through `tools/local_eval.py` with S14/S9 reference, inventory and dispatch audits, native NT1/NF math,
and Hopper attention stand-ins. All four final normal/forced screens pass13 correctness cases,
12dispatch shapes and24static checks each. `epi_next_screens.json` pins candidate hashes and coverage.
The complete payload passes all32local seeds, nodes and16phase seeds per M in82.8s (warm local builds);
`epi_next_lab_validation.json` pins payload642d6ae0 and exact output. `weights=None` is exercised.
 The first TREE normal screen compiles all24 sources/checks uncached;
later screens reuse only byte-identical source/flags from that successful same-image receipt. Changed
CUDA strings still compile with the original gate. Python/reference/runtime checks always execute.

GPU0 OOM attempts are failed receipts, kept under `runs/screen/nt1_r2_*previous.log`; they are not passed
screens. The resumed driver runs one GPU job at a time and only reuses successful exact-hash receipts.

Decision rule: all four independent arms are queued regardless of reset ratios; the payload includes
both S14 and PAIR. Nothing is waiting on a reset outcome before it can be sent. After both receipts, advance only a margin-safe paired kernel win to
integration's many-seed whole-layer pair. If no arm wins versus S14 beyond A:A, close these recipes.

## Reproduce / integration

`screen_epi_next.py --resume --run-lab` runs the normal/forced full-layer screens and then the complete
payload locally in the judge image. Evidence JSON records exact candidate/payload hashes. Local timings
are diagnostic, not H200 evidence. `patch_s14.py --kind tree|four|local|four_local -o <candidate.py>` emits
an experimental cell, patch and complete flag map onto S14. No second-round patch is promoted.
The earlier experimental PAIR patch remains `epilogue_s14_pair.patch` `786a5ecb` onto S14 `f1c77ce6`.

## Local ranking and limits

GPU0 local32seed ratios versus PAIR (M8/16/32): TREE1.00038/1.00030/.99935,
FOUR.99988/1.00570/1.00703, LOCAL1.00706/1.00961/1.00507,
FOUR_LOCAL1.00639/1.01262/1.00952. A:A mean absolute drift1.159/1.415/1.687us.
These show no clear win beyond PAIR on sm120; they do not select H200 performance. In the M32
local trace, LOCAL's norm export/gather costs.160/.504us and final combine2.514us versus PAIR1.568;
the structural atomics removal has a real tail cost. The H200 batch tests whether its reduction of
contention pays that cost there. No new H200 bandwidth or speedup is claimed.

The32 local router seeds give issue-normalized S14 math28.5625/34.3030/45.4292 instructions per unique
16weights/lane atM8/16/32, scalar epilogues1.7656/2.1205/2.8083 before other costs. Routing is runtime
input, so the payload prints the same accounting on H200. The older H200 route mix's M32 math45.979
in `ISSUE.md` is a different16seed sample, not a contradiction.
