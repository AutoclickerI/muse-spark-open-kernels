# P5 compact early history

**Current verdict: nominate, not closed.** Under the current point-estimate
handoff, exact-S14 E4/OLD4 is0.997414, across-seed SE0.001244, n16. P4
receives it for composition/held-out six-load confirmation. A:A is uncertainty
evidence, not a per-shape significance gate. No new history payload is sent;
P4 owns composition. The former A:A/3SE rejection is withdrawn.

Receipt `402819c28279466d`, wrapper `0cec6706`, records all seven arms passing
all16 public whole-layer TASK gates (historically with20% margin) and replay
checks. E2/E1/L4 vsOLD4 are1.016159/1.040193/1.020247. Against A, point
winners are OLD4/E4/E2/L4 at0.978186/0.975761/0.994125/0.997971,
SE0.014096/0.014496/0.014870/0.014298. A2/A1.009555,
SE0.006105, range0.98824-1.09546. Full samples, seed IDs and all estimates
are in [point_handoff.json](../sliding_round2/point_handoff.json).
[reset_result.json](reset_result.json) is regenerated under the point-handoff
rule; immutable receipt sources preserve the original experiment.

Preload implementation is proven: QT4 transpose median/max4.064/4.608us
precedes dependency-wait completion5.152us. Shared memory falls53504→28928B.
E4 attention increment12.160 vsOLD4 11.200us, QKV/gate7.296/1.536 vs
8.064/2.464us; the total's cause is not isolated. An early source load alone
would not have established the measured preload ordering.

`history_build.py` generates compact early shader0e07854b; `build_lab.py`
embeds exact S14 f1c77ce6, exact OLD4 shader3815beab and early/late arms.
`patch.py` is the integration generator; its former SHA7ce4f7a6 identifies the
historical generator. `lab_body.inc` contains capture/checker code. New labs
use the exact public TASK checker and retain route/output drift diagnostically;
old source receipts remain immutable. Extra tolerance margins and route
identity do not decide correctness or handoff.

`check.py` checks fresh/strided/partial/counter behavior, `sass_audit.py`
checks native instruction order/resources, `screen.py` provides normal/forced
layer screens, and `check_lab.py` runs the exact payload with forced glue.
Outputs go under ignored `out/`. Original local admission and hashes remain
in `admission.json` where present; no local timing establishes an H200 gain.

The prepared `../sliding_round2/followups.py history --selection '{"qt":4}'`
uses exact reset early/OLD4 shader bytes. It is an available experiment tool,
not a queued sender or a second promotion gate. Current P4 handoff/decision
rules are in [sliding_round2/README.md](../sliding_round2/README.md).
