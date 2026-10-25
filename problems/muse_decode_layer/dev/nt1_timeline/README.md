# P3a small-M MoE: current work

Base S14 final `f1c77ce6`; flags `{1:95,8:69,16:69,32:69}`. No new kernel promoted.
The reset batch is unchanged. Its result is not required to send the independent second batch.

## Reset queue (unchanged)

- MoE sender PID3360739, wrapper `917e79f37f28b5c1`, payload `prof/lab/moe_nt1_epilogue.py` `2b0356d7`.
- Not before **1792972800 =2026-10-26 09:00 KST**; priority1.0; no judge ID yet.
- `runs/waiting/3360739.json`, log `runs/detached/917e79f37f28b5c1.log`.
- EPBASE/VEC/PROD/NORM/PAIR plus fixed-math ONCE contrasts;16seeds, exactS14 control.

## Second-round batch (local screen in progress)

`build_epi_next.py` derives four cuts from the queued two-writer PAIR cell:

| Arm | Cell | SHA256[:8] | New work |
|---|---|---|---|
| TREE | `epiu_cell.py` | `30e50fdd` | Eight-way FP32 sum uses a balanced tree: native marked FADD chain8->3 |
| FOUR | `epif_cell.py` | `780b3cee` | Four16-lane writers instead of two32-lane writers, same modeled32 reduction bank wavefronts |
| LOCAL | `epil_cell.py` | `c9ce6f15` | Shared per-CTA norm sums, transpose export, parallel final gather; zero global norm atomics |
| FOUR_LOCAL | `epix_cell.py` | `b669c375` | Four writers plus disjoint local half-norm slots and final gather |

All retain Ksplit8, two packed stages and original dequant/MMA. No widened prepare weights.
LOCAL trades atomics for2KiB shared atM32,270KiB of current-call global export and a finalizer gather.
`ISSUE.md` gives the address/dependency model and its limits. `issue_audit.py` compiles marked actual
bodies sm90a; `issue_budget.py` normalizes per16weights/lane against4TB/s; `reduction_depth.py` checks
actual add dependencies. `audit_epi_next.py` confirms every full native flags69 image has128HMUL2/32HMMA;
LOCAL variants contain zero FP32 global atomic sites. Native performance regs125/125/125/127, zero spills.

`build_epi_next_lab.py` -> `prof/lab/moe_nt1_epi_next.py`: all four arms with S14 and PAIR controls,
32weight/router seeds, M8/16/32, readflush AB/BAx6, direct PAIR ratios, A:A, CUPTI nodes/unique-byte rates,
16phase seeds at each M, observer tax/effectiveGHz, local export/final gather clocks and route issue model.
Synthetic weights; independent of the first reset outcome. All arms fit in one second-round run.

Each arm passed528 component/replay checks +6forced ABI launches in the judge image (`check_epi_next.py`),
including real/empty/one/all8 routes, zero inputs,8/16/24/32-token edges, three local-start offsets and fresh
captured inputs. Worst TASK L2 .0024375; max relative versus INC .000251. Full normal/forced screens run
through `tools/local_eval.py` with S14/S9 reference, inventory and dispatch audits, native NT1/NF math,
and Hopper attention stand-ins. The first TREE normal screen compiles all24 sources/checks uncached;
later screens reuse only byte-identical source/flags from that successful same-image receipt. Changed
CUDA strings still compile with the original gate. Python/reference/runtime checks always execute.

GPU0 OOM attempts are failed receipts, kept under `runs/screen/nt1_r2_*previous.log`; they are not passed
screens. The resumed driver runs one GPU job at a time and only reuses successful exact-hash receipts.

Decision rule: queue all four second-round arms after local gates pass, regardless of reset ratios; the
payload includes both S14 and PAIR. After both receipts, advance only a margin-safe paired kernel win to
integration's many-seed whole-layer pair. If no arm wins versus S14 beyond A:A, close these recipes.

## Reproduce / integration

`screen_epi_next.py --resume --run-lab` runs the normal/forced full-layer screens and then the complete
payload locally in the judge image. Evidence JSON records exact candidate/payload hashes. Local timings
are diagnostic, not H200 evidence. `patch_s14.py --kind tree|four|local|four_local -o <candidate.py>` emits
an experimental cell, patch and complete flag map onto S14. No second-round patch is promoted.
The earlier experimental PAIR patch remains `epilogue_s14_pair.patch` `786a5ecb` onto S14 `f1c77ce6`.
