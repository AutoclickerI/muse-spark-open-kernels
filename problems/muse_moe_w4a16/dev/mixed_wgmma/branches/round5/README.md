# Fixed-N32 shared raw / shared decode

Follow-up to the valid topology loss `8792099636654bac`: ordinary launch and deeper serial
staging do not rescue full-expert native-wide RS. This tests a different ownership/cost split,
with **S14 and CTA2** direct controls and two new arms in one payload.

Payload **b610039e**, CUDA **be1f4121**, exact CTA2 prefix **5aff055b**. Rebuild with
`python3 problems/muse_moe_w4a16/dev/mixed_wgmma/branches/round5/build.py`; it regenerates its
retained parent dependencies. The generated `tokpair_payload.py`, adapters and CUDA are ignored.

Each256-thread CTA owns one64-output tile and two adjacent32-token groups. Register accumulators
remain N32. Both WGs share each packed Q/scale copy; each gathers only its valid current X rows
through SLOTS (W13) or ACT (W2), with no timed packing/gather kernel. All stages useS2.

- **RS_RAW_SHARE:** both WGs decode the shared raw packet into private A registers and consume
  RS-WGMMA. This isolates raw-copy reuse and ownership without a decoded BF16 round trip.
- **SS_DECODE_SHARE:** WG0 decodes once into a128B-swizzled BF16 A matrix; both WGs consume it
  through SS-WGMMA with their separate X slices. Both WGs also consume, with no dedicated
  producer WG, mbarrier full/empty protocol, dynamic register redistribution or global queue.

NT1<=8 and N16G4 for9-16 tokens, live routing, SwiGLU rounding, SS64 norms and F4 are unchanged.
No result is promoted before paired whole-MoE and whole-layer gates.

This differs from `9cd95e72`'s losing SS producer/consumer form: fixedN32 accumulators, two
consumer WGs also producing the data, one shared raw packet and no separate producer/full-empty
ring. It differs from TRANS64 by using direct X copies rather than register-fragment packing,
and from round4 by keeping two-CTA resource admission instead of64,512 registers perCTA.
The decoded-smem cost remains real and is the subject of the SS-vs-RS comparison.

## Mechanism model

On the fixed fixture, heavy32-token groups number5/11, while64-token groups number4/8.
Raw-copy weight multiplicity changes1.25/1.571 ->1.0/1.143. At4TB/s this removes at most
7.08/21.23us of scheduled weight service. SS decode work changes to0.8/0.727 of S14's;
RS performs1.6/1.455 as much decode because both WGs execute padded32-token groups.

Direct global activation reads remain unchanged: invalid rows issue zero-fill copies and do
not read global X. Padded copy/MMA instructions still increase. This distinction matters:
the earlier padded-X byte tables count instruction geometry, not actual requested global bytes.
SS additionally writes402.65/805.31MB of decoded BF16 A across all heavy CTAs, and the tensor
cores read A from smem twice. The service cost is unmeasured; do not convert the raw-byte saving
directly into an end-to-end speedup or assign the old overlap credit. This is a cost diagnostic,
without a numeric native latency forecast before measurement.

The native resource audit gives RS W13/W2 registers122/123 and43,008B smem; SS128/122 registers
and75,776B smem, zero spills/stack. Both permit two CTAs perH200 SM by registers and smem.
There are no C7512/C7513/C7519/C7520 warnings; wait1 and terminal wait0 are retained.
The shared-A store mapping is a bijection over all64x64 BF16 cells; the local SS stand-in
loads that actual matrix to matching mma.sync fragments. Native SS descriptor semantics and
actual Hopper residency remain unverified until the receipt.
Contracts: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).

## Screens

Both forced `tools/local_eval.py --tests-only` adapters:5 official tests and5 static checks pass.
[validation.json](validation.json):1,036 checks on128 captured input transitions, sizes33/127/
128/255/256/257, all hot/empty/real/count-class boundaries, poisoned ACT/Y/SS/OUT and12 shifted
local_start64 checks. All new outputs bit-equal to S14 on128/128 cases; worstL2/row0.004212/0.004782.
[entry.json](entry.json):24 seeds,72 fixtures/M/arm, real1GiB cold-read callback,150s deadline;
finishes71.0s locally with all gates passing, worstL2/row0.004132/0.005018. No Hopper performance
claim: local SS substitutes mma.sync and fits only one75KB CTA perSM rather than Hopper's two.

## Dispatch and decision

Sender **f0c92b1867e9aa50**, PID **127001**, job **7ea5e070b8f744ea**,
`muse_attention_decode` import host, budget150s, priority1.5. Exact metadata: [queue.json](queue.json).
This was sent on the stage3 loss; it is independent of the pending CTA2/queue/warp-specialized
outcomes because both S14 and CTA2 are direct controls. No earlier accepted run is duplicated.

Use SS/RS direct S14 and CTA2 pairs plus A:A and TASK gates. An SS win over RS isolates the
net benefit of sharing decode after the added smem/SS cost. An RS win alone leaves a cheaper
BF16 publication path open. Both losing closes these two fixed-N32 pair recipes only; retain
the best existing S14 or an independently verified round2-4 winner. Promotion requires P4's
whole-layer verification; a failed/incomplete control is unmeasured.
