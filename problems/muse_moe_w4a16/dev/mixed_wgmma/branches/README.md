# Light/heavy expert branches (M=128, M=256)

Serves G1 through the decode layer's M=128/256 MoE. One kernel family (`gemm.cu.inc`, namespace `mwbr`):
u8-scale recoded WGMMA/mma with typed `__align__(1024)` smem, N8/N16/N32 tiles, two selection classes
(FM=1 expert token count, FM=2 rows in the expert's 32-row group), a one-CTA live route (all id loads in
flight, no fills; META adds the heavy-only T32 and 8<count<=16 T8 lists) and the distributed finalizer.
`sources.py` adds H_H's NT1 kernel as `pmz` (small W13 zeroes its SS rows; THR 16 or 8).

## Handoff (current)

`nf_cell.py` fd04d8c4a100d2181051a8ab385f6dffc62b6ad32bb3236794006685342d3500: N_full at M=128 and M=256 (live route; heavy
N32 W13->W2 launched first on the side stream || NT1 <=16 W13->W2; one join; own buffers; reads S7's H_H u8 base + NT1
banks). On S7R: `problems/muse_decode_layer/cand_r3_best7_s7n.py` 2c9de5e57b8ea617c1c913d3738e4d0a1130e7b5b88f2669a6c2ffadbb18651c
(`dev/s7n/`: 72/72 bit-equal to the timed arm at both M, normal + forced 13 PASS). S9/S10 take it from `dev/s7n/build.py`.
Next lever: dequant instructions per weight (heavy path ~71 SASS / 16 weights; N_full is 2.1x / 2.8x the active-expert
HBM floor at M=128 / 256).

## Landed: M=256, H200 job 110599ba6b6144d4 (send a45ff19d; zero_()-flushed, paired ratios only)

vs exact S6 mixed_parallel 347.13 us, 12 seeds x3 fixtures, all TASK PASS, all bit-equal to S6:
R_full 0.9227 (fork once: side = <=16-row groups N16 W13->W2, main = full groups N32 W13->W2), E_full 0.9275,
per-phase joins 0.955-0.960, one stream 1.016 (parallel branches are the whole gain), R8_full 0.9958,
live route 0.9883 (10.59 -> 6.62 us), typed-smem W2 0.9831 (sm_120 showed 22%; H200 0.6%).
Handoff: `problems/muse_decode_layer/cand_r3_best7_s7r.py` = S7 with M=256 on R_full (`dev/s7r/`): bit-equal to
S7's mixed_parallel on 72 fixtures (`dev/s7r/bitexact.json`), normal local_eval 13 PASS + 23 static.

## Landed: M=256, H200 job 14f942d57e764ebf (send 701024b9; judge-style read flush)

vs exact S6 mixed_parallel 342.80 us, 12 seeds x3 fixtures, all TASK PASS: N_full 298.68 (0.8701), B8_full 0.8768,
N_phase 0.9063, R_s6 0.9230, NR_full 0.9305, HH_u8 0.9424, B8R_full 0.9595, NRm_full 0.9725, HH 0.9941.
Handoff: `problems/muse_decode_layer/cand_r3_best7_s7n.py` = S7R with M=256 on N_full (`dev/s7n/`; mixed_parallel's
engine deleted). The M=128 half skipped on a 60 s pre-compile gate (now 25 s); resent as `49cd13f548734cd3`.

## Landed: M=128, H200 job c01b1916e2c74af8 (send 49cd13f5; judge-style read flush)

vs exact S7 H_H 234.78 us, 12 seeds x3 fixtures, all TASK PASS: N_full 207.40 (0.8842, bit-equal 36/36), B8_full 0.8812,
N_phase 0.9307, NR_full 0.9474, HH_u8 0.9702, B8R_full 0.9770, U16 0.9931 (S7's in-register word costs 0.7%), NRm 1.0025.
N_full is one cell for both M: `nf_cell.py` (heavy grid 1056 at M<=128, 2112 at M=256); `cand_r3_best7_s7n.py` routes
M=128 and M=256 through it.

## Landed: N_full tuning, H200 job 880c388117ed4e16 (send 2bb856ba)

vs N_full (M=128 207.60, M=256 299.49 us): heavy on the side stream (launched first) 0.9799 / 0.9882; priority adds
nothing; NT1 first at high priority 1.04; NT1 grid x0.75 / x0.5 1.02 / 1.07; heavy grid /2, x2 neutral. `nf_cell.py`
and `cand_r3_best7_s7n.py` now launch the heavy branch first.

## Built: `build.py` -> `prof/lab/moe_br_m{128,256}.py` (branch arms), `moe_nf_tune_m{128,256}.py` (BR_TUNE arms)

Controls: M=128 HH = exact S7 H_H (u8 `hh_cell.py`); M=256 S6 = exact mixed_parallel, plus R_s6 (= R_full).
U16 (M=128) = exact u16-word H_H: S7's in-register scale word cost. HH_u8 = H_H with this family's WGMMA.
Arms: N (NT1<=16 beside heavy N32), NR (heavy remainders on N16 beside NT1), NRm (remainders on the heavy
stream), B8 (W2 NT1<=8 + N16 for 8<count<=16), B8R; `_full` forks once, `_phase` per phase.
`check.py M` gates every arm (poisoned scratch, none/hot/remainder/single routes in one graph, nearby M,
local_start 64) then runs the lab: both pass locally (`check128.json`, `check256.json`).
MoE-host screen (`lab_screen.py`, 5 tests + 6 static) passes. Sent together: send `701024b91f79ac28` (MoE host).
