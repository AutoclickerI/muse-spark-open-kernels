# S5: H_H expert and phase mix at M128

Candidate: `../../cand_r3_best7_s5.py`; base S4 `9b2b2a48` is frozen.
`build.py` extracts the payload string embedded in the accepted H200 wrapper
for job `770809cf159f41e2`, SHA-256 `f3c2796944c60b77aa5d63fd61e766a386bcdcfaf526b8ed01ab19e5367e5736`.
`manifest.json` pins the candidate and exact native sections; `audit.py`
verifies them against that immutable accepted wrapper.

H_H wins on all36 gated fixtures (12 weight/router seeds x3 fresh inputs):
234.44 us mean versus D32g272.70 and NT1-recoded289.62. Mean paired H:D32
ratio0.8629; maximum L2/row errors0.00413/0.00469. Six rotated/reversed cold
128MiB graph orders and duplicate D32g A:A give mean absolute noise1.580 us
(max8.832). These are standalone H200 measurements, not S5 whole-layer time.

Each timed call runs the measured GPU route plus metadata kernel, which
splits experts at16/17 tokens. W13 and W2 each launch the one-LOP3 NT1 phase
for counts<=16 and WGMMA32 for counts>16, with disjoint and complete GPU task
sets. WGMMA uses which1/grid1056; NT1 uses two stages subject to device shared
memory and occupancy. A separate distributed finalizer normalizes and combines
all local routes. Both phases are fixed to H. All routing-dependent work stays
inside the timed call; launch count and host allocations do not depend on IDs.

Native idist, pmnt and metadata sections are byte-exact extracts from the
accepted source. Unused benchmark/control native sections are omitted. The
original `pm_plan`/`pm_call` definitions preserve all measured pointer owners,
allocations, launch order and current-stream lookup. The existing S4 bank
packing and lossless u16 scale descriptors match the payload, and the unchanged
recoded NT1 bank supplies the phase kernel. No prepared expanded weights are
retained. The replaced standalone D32 engine and dispatcher are removed.

Only the two M128 table cells change. Small-M one-LOP3 NT1, M256 C1g2112,
projections, attention, glue and other-shape policy remain S4's. H_H uses a
mandatory three-fixture FP32 prepare gate with six forward/reverse fresh-route
graph replays. A gate failure raises and stops admission.

Reproduce the screens and source audit:

```bash
python problems/muse_decode_layer/dev/s5/build.py
python problems/muse_decode_layer/dev/s5/audit.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best7_s5.py --no-baseline \
  --json problems/muse_decode_layer/dev/s5/normal_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python \
  problems/muse_decode_layer/dev/s5/force.py --cold-native
python problems/muse_decode_layer/dev/s5/write_validation.py
```

The forced screen requires eager and captured native H_H calls in both M128
rows, along with the retained native experts in the other10 scored rows.
Hopper-only attention/DG/Humming launches use the established stand-ins;
H_H executes its actual portable CUDA arithmetic. Native SM90a compilation is
checked separately. Both screens pass13 tests/12 scored graphs/21 static checks.
Prepare takes13.773 s normal /42.365 s forced, including0.310/0.333 s for the
M128 banks. H_H's fresh-route FP32 gate reaches L2/row0.003817/0.004310.
The source-pinned gate, dispatch and prepare evidence is in `validation.json`.
Whole-layer H200 composition and cold H200 prepare remain unverified.

The MoE lane's fixed cell landed during assembly:
`hh_cell.py` SHA-256 `970663d44d993f8a029cd4289e2a9716bd2bf5897098799a6e0605e0abf21f0f`.
The audit cross-checks all three native bodies against its frozen handoff.

S5 is queued scored-only at priority0 behind S1–S4 for2026-10-25 09:01 KST:
`runs/detached/f7064fc0127e80a9.log`. The candidate source SHA-256 is
`f7064fc0127e80a955c7dc9aa5c68eb6aefada179c2841c5c58bce8dabd20284`.
`queued.json` verifies the frozen snapshot and all four prior queue entries.
Refresh from landed S1–S4 profiles with
`python problems/muse_decode_layer/dev/levers/build.py`; this also updates PLAN
Status from the latest cumulative profile and applies each measured correction
once. No watcher or judge polling is added.
