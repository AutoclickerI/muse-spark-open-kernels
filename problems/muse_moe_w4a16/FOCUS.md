# MoE focus
G1/G2: M128 MoE lane; M256 free lane raw385.2/rec365.2us in S2/S3. W13 lab8483c0c6 pending; serving parked.

Incumbent H_H: H200770809cf159f41e2,234.44 vsD32g272.70/NT1rec289.62us;
paired ratio0.8629 vsD32g,12 seeds x3 fixtures, all gates pass,L2/row.00413/.00469.
Both W13 and W2: one-LOP3 recoded NT1 for count<=16; exact-u16-scale
packed WGMMA32 config1/grid1056 above16; common routing, distributed finish.
Seed0/fixture0 cold route/W13/W2/final10.1/132.9/84.9/12.8us; A:A abs1.58us.
H_D32238.89, N_D32263.21,N_H258.50,N_N269.12,one-read362.43us.
Mechanism supported: expert-size tradeoff, repeated NT8 decode vs padded
WGMMA/transport/address work. One-read alone loses; no unique stall counter.
W13~133 vs~75us unique-byte model remains the main target; W2~82–85us.

Fixed H_H handed to decode FOCUS for S5; actual wrapper25 graph gates pass:
dev/mixed_wgmma/m128/phase_mix/hh_cell.py
SHA256 970663d44d993f8a029cd4289e2a9716bd2bf5897098799a6e0605e0abf21f0f
hh_prepare(weights); hh_experts(x,scores,local_ids,state,local_start=0).
FINDINGS tells the free/M256 lane the full H_H recipe and evidence.
Whole-layer H200 composition remains unverified.

ONE proj-host snapshot scheduled:d8d3a89cb3436100, release10-25 09:01 KST.
Detached log inspected; held outside queue; no new judge job before release.
prof/lab/moe_m128_mix_sweep.py; proof dev/mixed_wgmma/m128/mix_sweep/.
25 new recipes +exactH_H: thresholds8/16/24/32,heavyN32/N64 independently
per phase and coupled,W2-all-NT1/all-WGMMA controls.12 seeds x3 fresh routes,
six cold128MiB AB/BA orders/A:A; component phase means cover12 seeds.
Local936 graphs +575 threshold/signed/zero/subnormal/scratch gates pass;
13 sm90a entries no spills/C7513; proj9 tests/30 graphs/3 static pass.
Prepare19.83s,payload17.36s. Send-time gate tested without API calls or sleep.
Next resume: inspect scheduled log/receipt once; use complete paired means.
