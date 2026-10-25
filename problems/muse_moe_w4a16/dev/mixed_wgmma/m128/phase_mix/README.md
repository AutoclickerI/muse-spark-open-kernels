# M128 phase and expert composition

Serves G1/G2: lower MoE time inside the decode-layer score. H200
**770809cf159f41e2 is complete and H_H wins**:234.44 us vs D32g272.70 and
NT1-recoded289.62 across12 seeds x3 fixtures, paired ratio0.8629 vsD32g.
All36 numeric gates pass; maxL2/row0.00413/0.00469, A:A abs mean1.580 us.
The fixed H_H handoff is [hh_cell.py](hh_cell.py), SHA-256
`365b623992367a1d45d3dae06a2096ddd0e6879deffab2e3924678a47997b621`: uint8 scale codes,
BF16 scale word formed in registers (the S7 kernel; bit-identical to the u16-word cell 770809cf timed).
`hh_prepare(weights)` then `hh_experts(x,scores,local_ids,state,local_start=0)`
at M128. Both phases use one-LOP3 recoded NT1 for count<=16 and packed
WGMMA32 config1/grid1056 above16, then distributed per-token finalization.
The actual fixed wrapper passes25 signed/zero/subnormal/separate-global and
changed-route graph gates; decode FOCUS has the path/hash for S5. No whole-layer
H200 gain is claimed yet. `build_hh_cell.py` reads the accepted payload and
retains exact native kernels, packing, recipe, lifetime and stream behavior.

Completed MoE-host lab: send **7d9404dc53d78d6f**, judge
**770809cf159f41e2**. `h200_result.json` pins all measured values. H_D32238.89,
N_D32263.21, N_H258.50, N_N269.12 and OneRead362.43 us. H_H seed0/fixture0
cold graph route/W13/W2/final phases10.1/132.9/84.9/12.8 us; H_D32 W2 is82.5
on that fixture despite losing the whole seed mean. Repeated-decode versus
padding/transport overhead depends on expert size and phase; no counter proves
one hardware stall. New threshold/N32/N64 and independent-W2 work is in
[../mix_sweep/](../mix_sweep/README.md), one proj snapshot**d8d3a89cb3436100**
runs as judge job `4c469bf9ef9645b1`. It measures component phase means on12 seeds.

The prerequisite one-read result **640d67e7e56d42e5** landed: NT1-recoded290.83,
RS64g528378.67, RS128g528425.03, RS64g2112361.16, SS64g528750.96,
SS128g528785.54, SS128g1056678.86 us, all36/36 numeric gates pass.
Best RS W13/W2/finish graph phases are235.4/124.6/12.6 us; SS remains much
slower. Thus one logical packed-weight read per expert does not by itself give
a faster implementation. Address/transport overhead and synchronization remain
relevant; no hardware DRAM counter or unique causal limit was measured.

The review's work-count claim was checked first. On the12 M128 rows in
`dev/v9_reuse/seed_sweep.json`, eight-token NT1 for counts<=16 plus WGMMA32
above16 reduces groups by **35.42% mean,25.81–48.65% range**. Extra padded
token work relative to NT1 is **11.74% mean,3.23–25.81% range**. Aggregate
ratios are35.54% and11.52%; padding relative to actual tokens is55.41% mean.
The stated35–50% reduction and12–14% padding are reasonable averages but
not per-seed ranges. `route_audit.py` reproduces all counts with the input hash.
These counts do not measure bytes served by HBM or wall-clock time.

All new compositions use one timed route layout: C/OFF/SLOTS/POS with expert
rows padded to32. A GPU metadata kernel builds separate prefixes for NT16,
large-expert NT32, and expert-wide NT128, and zeroes norm partials. No host
branch, allocation, synchronization or launch count depends on routed values.
The existing distributed finalizer reads the common Y/SS layout. Every phase
launch retains its tensors and resolves the current stream.

`N` is a standalone recoded NT1 phase with the already-admitted one-LOP3
unpack. It reads common route metadata instead of re-deriving token lists in
each CTA. W13 keeps BF16 gate/up and activation boundaries and independent
global scales. W2 applies PNW and writes norm sums into64 common partials.
`H` launches N for experts with<=16 tokens, then packed WGMMA32 for>16;
the GPU task prefixes make the sets disjoint and complete. `D` is WGMMA32,
config1/grid1056, `D16` config0/grid1056, `Dlow` WGMMA32/grid528, and `R`
is the best recoded one-read RS config3/grid2112. Both phases may select each
backend independently. The R path still reads each expert tile once; its rows
now use the common32-padded layout and independently built128-token tasks.

| Lab row | W13+SwiGLU | W2+norm partials |
|---|---|---|
| D32g | Exact accepted D32g allocation/route/phase path | Exact accepted D32g |
| NT1rec | Exact accepted fused NT1-recoded control | Same fused control |
| OneRead | R | R |
| N_D32 / N_D16 / N_Dlow | N | D / D16 / Dlow |
| H_D32 / H_H | H | D / H |
| N_H / H_R | N / H | H / R |
| R_D32 / N_R | R / N | D / R |
| N_N / D16_D32 | N / D16 | N / D |

D32g's174.2/81.0/12.4 us are separate cold graph phase medians on seed0,
fixture0. NT1's130.7 us W13 is an instrumented per-CTA median, not a standalone
phase wall time; its maximum is161.8 us. They are not additive or directly
interchangeable. The new lab measures real standalone N and composed phases.

Protocol:12 weight/router seeds x3 fresh real routes, identical inputs for all
14 rows; six rotated/reversed AB/BA graph orders, cold128MiB before each replay,
duplicate D32g A:A, independent FP32 finite/L2<=1%/row<=5% gates before and
after timing. Each fixture uses the median of each variant's samples; reports
give arithmetic means, means of paired ratios, seed lines and A:A spread.
Phase graphs refresh prerequisites before their cold replay. All fixtures must
complete; partial or failed results cannot promote a path.

Local evidence: **504 graph checks/36 fixtures**, maxL2/row0.004131/0.004847;
**220 edge checks**, max0.004104/0.004601, including the exact D32g wrapper.
Edges stress0/1/8/15/16/17/31/32/33/64/65/80/128 tokens, simultaneous16/17
experts, signed/zero/subnormal scales, separate gate/up globals, local_start64,
allocator churn and complete finite active ACT/Y/SS writes after poison.
All22 SM90a entries have no spills or C7513; four new NT1 phase kernels use
109–112 registers. Admitted D32g and NT1rec controls match accepted source
bytes. Host5 tests/6 scored graphs/6 static checks pass, prepare18.35s and
payload14.67s within85s. SM120 executes immediate MMA for WGMMA but the same
route, bank, scratch, pointer and finalizer glue. Local times do not rank H200.

Reproduce with `build.py`, `tools/h200_lab.py build` on the MoE host, then
`tools/img python native.py` and GPU0 `tools/img env MAX_JOBS=2 python check.py`
and `check_edges.py`. H_H is handed off above. The next receipt belongs to the scheduled proj-host
threshold sweep; use its complete paired mean and component phases.
