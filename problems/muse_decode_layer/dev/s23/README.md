# S23 integration

S23 = S22 phase 1 (`fd5e0ca2`, shipment `22874960`, G0 R 1.009997 [1.009669, 1.010326] over S21 on fresh seeds, job
`0e1f9bc8`) + P10's held-out projection ports + the P16 pool/router carrier, with the d32S direct-input hybrid
disabled. Source `b939b927`, shipment `problems/muse_decode_layer/cand_r3_best7_s23_final.py` (`b2d9a6dd`).

| Group | Change | Leave-out rows | Evidence before S23 |
|---|---|---|---|
| PORT2 | T1 unpooler_slice, PDL gemv (P) | d1S, d1F | P10 parent vs S21 on 8 held-out seeds (`72f273a8`): 1.0071 [1.0052, 1.0091] for all ports |
| PORT4 | T16 qkv_gate, NF (N) | d16S, d16F | same |
| PORT6 | T16 gate, PDL cluster (P) | d16S, d16F | same |
| PORT11 | T256 qkv_gate, NF (N) | v32S, v32F | same |
| PORT14 | T256 unpooler, NF (N) | v32S, v32F | same |
| P16 | T16 pool/router, NF12 S4/WK4/ST1 DSM RED3 | d16S, d16F | P16 over CW8_S2 1.0021 [1.0002, 1.0040] (`72f273a8`) |
| HYd32S | d32S back to the S21 Lee port | d32S | S22 phase-1 hybrid 0.99874 [0.99783, 0.99965] (`0e1f9bc8`) |

P10 measured the ports on S21, where d16S was a port; S22 restores S19 at d16S, so the d16 leave-outs measure that
interaction. P10's patch builds its banks in `prepare()` wrappers; the paired lab prepares arms through the
`_s16_prepare_delta` chain, so `compose.py` appends the same helpers verbatim (`port_body.py.inc`, `night/body.py.inc`,
MIT notice included) and runs the same bank builds and warm calls as the last delta step, after the whole S22
prepare. Dispatch stays S22's until those banks exist, as P10's wrappers did during the base prepare.

Lab (`build_lab.py`): A = S23, A2 its twin, INC = S22 phase 1; one restricted leave-out per group (it runs only on
its rows and takes A2's time elsewhere, where its code equals A's). 64 fresh seeds (1203400011 + 104729 i, disjoint
from every earlier selection and confirmation), new weight/router seeds, 16 symmetric rounds, TASK on every arm
(3136) + 260 nearby. Group R = time(leave-out)/time(S23) over six loads: > 1 favors keeping the group.

Gate: S23 is scored only after S22 phase 1's official run passes, and only if its lower 95% bound over S22 phase 1
is above 1 (`tools/promotion.py`), with in-lab TASK and nearby complete and prepare <= 300 s. A group whose upper
bound is below 1 is dropped and the composition reconfirmed on new seeds.

Local checks: 25 contracts (`contracts.py`, including S22's base contracts for drop v16F + hybrid d32S off),
normal/forced stand-in screens 13 TASK + 12 graphs each (P10's `night/screen_s22.py`), exact packed rehearsal 3136
TASK + 260 nearby with every arm complete (`screen_lab.py`), lab-only transport. Wire 8938 bytes.

Sent: `lab_s23_pair.py` on muse_bf16_proj as a check, wrapper `3b4aff3b`, lane P4. Results: `results.json` after
`python3 problems/muse_decode_layer/dev/s23/report.py <receipt>`.
