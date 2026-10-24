# Decode-shape adapter lab

`build.py` builds the standalone `../moe_decode_adapters.py` from the verified
producer adapters in `moe_mixed_wgmma_prefill.py` and the controls/generator in
`moe_v9_seed.py`. Source hashes are embedded in the payload.

The comparison contains native V9 NT1/NT2, raw17, Humming GROUPED_MASKED and
DeepGEMM dg_w8 at M128/M256. R0 uses the exact captured IDs from the earlier
loaded-weight lab, with refreshed activations/route weights and seed5353 expert
weights. The generator track uses all twelve seeds5353+7919*i and the same three
fixture seeds9000+M+7919*j as the earlier V9 seed lab. The public weight/router
RNG is unchanged. A separate seed+1000003 generator sets upGS = gateGS *
exp(.55*N), giving distinct FP32 gate/up scales for every expert.

Only the scale handling changes from the earlier adapters:

- V9 and raw17 apply independent gate/up GS in their existing W13 epilogues.
- Humming imports unit W13 global scales, then the existing portfolio SwiGLU
  applies the actual distinct gate/up GS. Its importer otherwise folds distinct
  stacked GS into BF16 block scales. The launch config is still selected per
  shape, copied from the heuristics dict, and has a valid full M range.
- NVFP4 codes retain their values under layout repacks. Stored E4M3 scales retain
  their bytes. raw17 and DG convert them inside their timed kernels. DG expands
  a fresh scratch bank inside every call; no expanded bank is prepared or cached.

An independent FP32 oracle checks all three refreshed graph fixtures before a
variant can be timed: whole-output relative L2 <= .01, max token-row relative L2
<= .05 and finite values. Each of eight cold128MiB timing rounds replays all
passing graphs on the same fresh fixture in rotated/reversed order. Per-seed
rows record all five times and errors; each variant's arithmetic seed mean is
reported independently with a count out of twelve. Rejected rows never enter a
mean, and partial means are labeled incomplete. Times include graph overhead,
matching the earlier V9 seed lab.

Local validation: run `CUDA_VISIBLE_DEVICES=0 tools/img python
prof/lab/decode_adapters/screen.py`. It exercises all R0/public-seed graphs with
real V9/raw17/Humming arithmetic. DG's Hopper-only Python ABI is forced through
an explicit zero-output stand-in; its actual per-call W13/W2 expansion is
separately compared bit for bit with independent dequantization. Stand-in output
is never presented as a DG arithmetic or timing result. The H200 lab imports the
real DeepGEMM and applies the same FP32 gate to it as to the other four variants.

Local screen passed 130 variant/shape rows: 104 real arithmetic gates (26 each
for NT1, NT2, raw17, masked Humming) and 26 DG stand-in ABI checks. Maximum
whole-output L2 was 0.005793 and max token-row L2 was 0.007911. The three CUDA
strings pass SM90a compilation; undefined names, call signatures and judge-image
imports pass. H200 arithmetic/timing remains unverified until the lab receipt.

Queued on muse_moe_w4a16 at priority 0.5, using h200_lab.
Payload SHA256:5b157768b0621880d0e48c72654aff0168a4c688f29046ba43a4fa18c1ab7052.
Submitted source SHA256:85550e903b56d3604c72b3f11faeb637edc338628832f4fa17612ad6625b52c7.
The submitted source key is 85550e903b56d360; it is not a job ID. At handoff
the detached sender is waiting for the team's inflight cap, so a judge job ID
has not been assigned. Its acceptance line will be written to
`runs/detached/85550e903b56d360.log`.
