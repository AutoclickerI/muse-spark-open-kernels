# P4: S12 integration

Base: `cand_r3_best7_s11_final.py`, SHA256 `acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7`.

The build pins every incoming patch's hash in `pieces.json`, asserts the base, and records each intermediate
hash in `manifest.json`. `--without <name>` builds each leave-one-out arm. No scored send from this lane.

Incoming pieces:

- `coded`: qkv_gate and pool_router at T=1/8, patch `problems/muse_bf16_proj/dev/coded_dense/patch_coded_s12.py`.
  Standalone H200 `320884442a6f4624`: qkv 10.74/11.20 vs balanced 11.66/11.78 us; pool_router 18.75/19.14 vs
  21.12/21.33 us. Waiting for dense lane's normal/forced S11-final screen and whole-layer S12 A/B.
- `final4`: received `patch_s11_f4.py` ef041fd4 on S11 final's nf cell. Standalone H200 `5e197b5eb7424a1e`:
  F4/LI_swap 0.9850 at M128, 0.9885 at M256; bit-equal.
- `wide`: received `patch_s11_wide.py` b2748e23; H200 `f976b09d3fc445eb`, LIW/LI4 0.9936 / 0.9900 at M128/256,
  12 seeds x3, all TASK PASS. Separate arm and independently prepared original-layout bank for its reversion.
- `coresidency`: excluded after `f976b09d3fc445eb`: G1 1.2351/1.4045, G2 1.0242/1.0586, G3 1.0285/1.0433,
  G1p5 1.1951/1.3436, G2p5 1.1115/1.0882 at M128/256. Control 191.79/270.98 us; LIW 190.22/268.18 us;
  A:A abs 0.565/0.440 us. LT2 0.9993/1.0015 is neutral.
- Small-M, glue, attention: only patches with a paired H200 win.

Whole-layer protocol: 16 seeds x 8 rotating rounds; cold L2 before every replay; A:A; each piece removed;
all S11 back. Promotion requires complete normal/forced local screens, precision inventory, all 12 H200
shapes and six weighted loads, no failed correctness arm, S12 better than S11 at every load, no losing piece.
If the fixed pieces leave an unchanged load, its S11 arm is an A:A control, not evidence of a speedup.

Current candidate: `cand_r3_best7_s12.py` **7d4216dd**, three frozen patch snapshots in this directory. The
dense lane fixed its original pointer-keyed global code bank: the admitted patch ccb512c3 stores codes in state,
so judge tensor clones retain valid weights. Gate split bypasses coded qkv at T=1; pool_router still runs coded
there, while T=8 uses both coded roles. The first (coded-only) local attempt exhausted GPU0 memory amid concurrent
lane screens and was discarded. The assembled candidate passed normal 13 tests and 12 graph captures;
all 23 static checks pass. Forced screen is pending. [Precision audit](PRECISION.md).

Local wrapper admission found a fixture-generator issue before sending: the candidate's embedded public
`_ap_generate_input` calls FA3's Hopper scheduler for full decode inputs even on sm120. The lab now does what
local_eval does for its reference: suppresses only fixture scheduler metadata off Hopper. H200 still calls the
original scheduler. It also carries unchanged runtime/cache assignments into independent arm namespaces,
without sharing function bindings or mutable GEMV-choice dictionaries.

Next: finish normal/forced screens, exercise the full lab, then send the check. Preserve S11 final for the
orchestrator's 00:00 UTC scored send until an S12 final is verified and explicitly frozen here.
