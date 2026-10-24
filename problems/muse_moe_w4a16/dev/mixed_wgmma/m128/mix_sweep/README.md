# H_H threshold and phase sweep

Serves G1/G2: lower M128 MoE time inside the decode layer. The incumbent is
the fixed H_H cell from **770809cf159f41e2**,234.44 us vs D32g272.70 and
NT1rec289.62 (12 seeds x3 real-route fixtures, all gates pass). H_H uses
recoded NT1 for expert counts<=16 and packed WGMMA32 config1/grid1056 above16
in both W13 and W2, plus the distributed finalizer. Its S5 handoff is
`../phase_mix/hh_cell.py`, SHA-256
`970663d44d993f8a029cd4289e2a9716bd2bf5897098799a6e0605e0abf21f0f`.
Decode FOCUS links the fixed cell; its actual wrapper passes25 graph gates.

The measured result supports an expert-size tradeoff between repeated NT8
decode/loads and WGMMA padded compute, transport, and address work. It does
not identify one hardware stall. The one-read producer loses, and switching
all experts to either backend also loses. H_H seed0/fixture0 standalone cold
route/W13/W2/final phases are10.1/132.9/84.9/12.8 us; H_D32 W2 is82.5 on
that fixture, yet H_H wins the full seed mean by4.45 us. Per-seed phase evidence
is needed. W13 is still roughly1.77x the~75us unique touched-W13 byte model.

This single proj-host lab sweeps thresholds8/16/24/32 and heavy WGMMA32/64
in each phase independently, then coupled. It adds W2-all-NT1, W2-all-WGMMA32,
and W2-all-WGMMA64 controls: **25 experimental recipes plus exact admitted
H_H**. Recipe names `A{threshold}n{N}` sweep W13 with W2 fixed16/32;
`B` sweeps W2 with W13 fixed16/32; `C` uses the same recipe in both phases.
Duplicate16/32 recipes are printed once. A16n32 checks the new metadata and
runtime-threshold transport against the byte-pinned admitted H_H control.

All tensor formats, packing, BF16 activation boundaries, PNW/norm/finalization
and seed inputs match the admitted cell. The new NT1 phase reader changes only
the expert-count cut to a launch-constant argument. GPU metadata builds disjoint
heavy task prefixes for each phase. N64 instantiates the existing exact WGMMA
producer with NT64/KT128/S4, versus NT32/KT128/S4. The route layout remains
32-padded; no output for inactive rows is written. All raw pointers retain their
owners and use the current stream. There is no input-dependent host launch
count, allocation or synchronization, and no prepare-expanded expert weights.

Proj lacks expert weights, so the payload creates the same public synthetic
MoE/router weight seeds5353+7919*i as the admitted MoE-host lab. It uses12
seeds x3 fresh generated real routes, identical across all26 rows. Six cold
128MiB rotated/reversed AB/BA graph orders and duplicate H_H A:A samples
feed per-fixture medians, arithmetic seed means, and means of paired ratios.
Independent finite/FP32 L2<=1%/row<=5% gates run before and after timing.
Every component is separately graphed on the first fixture of **every seed**:
route, W13-small, W13-heavy, W2-small, W2-heavy, finish. Prerequisites refresh
before each cold replay. Phase means cover12 seeds; the whole-call graphs
choose winners. Separate cold phases are not additive whole-call latency.
An inactive side in an all-NT1/all-WGMMA control is an empty graph; its event
overhead is not kernel work. Partial seeds do not qualify for promotion.

Local validation: **936 graph checks/36 fixtures**, maxL2/row0.004130/0.004828;
**575 edge gates**, max0.004102/0.004601. Edge counts straddle every threshold
and N64 boundary:0/1/7/8/9/15/16/17/23/24/25/31/32/33/63/64/65/80/128,
with all cuts simultaneously active on other experts. Signed/zero/subnormal
scales, separate gate/up globals, local_start64, allocator churn, exact task
prefixes, and complete active ACT/Y/SS writes after poisoning pass. All13 new
SM90a entries have no spills/C7513; N32 and N64 HGMMA shapes are audited.
Native W13/W2 registers: N32 108/105, N64 122/138. Runtime occupancy is
reported by the lab on H200, not inferred from the local GPU. The actual proj
host wrapper passes9 official tests,30 scored graphs and3 static checks
(prepare19.83s).
The full payload completes locally in17.36s within105s, with12 phase seeds.
SM120 uses immediate MMA to execute the same bank/shared/scratch/route glue;
its times do not rank H200.

The owner requested one lab **after09:00 KST** on proj. `--not-before
2026-10-25T09:01:00+09:00` freezes the snapshot now, keeps it outside dispatch
order, and prevents service reads/sends until release. Other lanes' ready labs
are not absorbed into this delay. `check_send_gate.py` verifies this with a
simulated clock and fake sender, including timezone handling and queue order;
it does not sleep or contact the service. Actual frozen identity: **d8d3a89cb3436100**, held until **2026-10-25
09:01 KST**; detached log inspected. The queued wrapper matches the fully
screened file byte-for-byte. Its identity and log are in `sent.json`. A scheduled key is not a judge job id; the latter exists only
after release and judge acceptance. No new H200 performance is claimed.

Reproduce: `build.py`, build on the proj host with `tools/h200_lab.py build`,
then `tools/img python native.py` and GPU0 `tools/img env MAX_JOBS=2 python
check.py` / `check_edges.py`. On resume inspect the single scheduled log/receipt
once and act on complete paired means. Emit a fixed winning cell for decode
only after its H200 verdict.
