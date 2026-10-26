# early history

Early sliding-history gather/transpose experiment on exact S14; the compact
recipe did not beat its control beyond noise on the H200.

Files (one entry per retained direct file):

- [.gitignore](.gitignore): Generated-output exclusions.
- [README.md](README.md): Current working set and resume instructions.
- [build_lab.py](build_lab.py): One immutable attention candidate with one batched exact-S14 whole-layer history lab.
- [check.py](check.py): Fresh data, strided producer partials and repeated captured counter reset.
- [check_lab.py](check_lab.py): Execute the exact batched payload locally with forced Hopper glue and two fresh seeds.
- [history_build.py](history_build.py): One-tile sliding history, retained before PDL wait; compact shared-memory lifetime.
- [lab_body.inc](lab_body.inc): Device source or payload template used by retained code.
- [patch.py](patch.py): Experimental compact early-history last-CTA on exact S14 final.
- [sass_audit.py](sass_audit.py): Rebuild sm90a artifacts and retain bounded instruction-order/resource evidence.
- [screen.py](screen.py): Normal and forced-Hopper whole-layer glue through local_eval, actual compact tile math on sm120.
- [out/p5_early_history_board.py](out/p5_early_history_board.py): Muse verify stream portfolio: one paged KV pass per split and KV head.
- [out/s14_early_history.py](out/s14_early_history.py): Runtime or source-generation dependency.

Handoff: [patch.py](patch.py), SHA256 `7ce4f7a6cb29bd00d87ece330c6b72b15d954a721b56a73af0b484381ba03ba8`, applies onto S14 f1c77ce6.
Base source: [S14 final](../../../muse_decode_layer/cand_r3_best7_s14_final.py), SHA256 `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

## Native decision

Wrapper `0cec6706` completed as **402819c28279466d**. All seven arms
passed all16 public whole-layer TASK gates with20% margin and the replay
drift checks. Best E4/OLD4 is **0.997414**, seed SE0.001244; E2/E1/L4
are1.016159/1.040193/1.020247. A:A is1.009555, SE0.006105,
seed range0.98824-1.09546; the original-control comparison is noisy.
The0.26% incremental E4 gain is below both the A:A displacement and three
standard errors. The prewritten decision therefore drops this recipe:
**no history confirmation sent and no new P4 promotion**.

The implementation is measured, not an unproven preload: QT4 history is
transposed by median4.064us (max4.608), before the dependency wait completes
at5.152us. Shared memory drops53504→28928B at138 registers.
Nevertheless, E4's attention completion increment is12.160us vs OLD4's
11.200us; QKV/gate increments are7.296/1.536 vs8.064/2.464us.
The whole-layer pair, rather than shifted stage boundaries, decides.
QT2/QT1's epilogue tails are3.136/5.248us; compact early history alone
does not solve those tails. These phase clocks are instrumented diagnostics.

This closes only the compact single-tile, last-CTA early-history recipe at
decode1 sliding. Resident-history verify and round2's cooperative/query-row
rebuilds remain independent tests. Machine audit: `reset_result.json`.
The held-out history generator now preserves exact OLD4 CUDA3815beab at
QT4 as its control; it remains unsent because this decision failed.
