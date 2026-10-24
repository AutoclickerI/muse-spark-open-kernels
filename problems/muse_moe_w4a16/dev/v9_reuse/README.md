V9 tile reuse, pending H200 receipts

`dropin.py` is self-contained. Call `moe_reuse_prepare(full_decode_weights)` once,
then `moe_reuse_experts(x_bf16, scores_fp32, local_ids_int32, state)` per step.
Local IDs use -1 for remote experts. Replace the old expert bank's preparation;
old E5M3-reencoded scale banks are incompatible with this raw-E4M3 bank.
`cand_r3_v9_reuse.py` in the decode problem is the complete-layer integration.

The prepare gate uses three synthetic fixtures and independent FP32 expert math,
then six alternating cold graph pairs. Only M128/256 select new configurations.
The full-K shared mode requires H200; local native arithmetic covers register and
K2048 shared modes, and a portable stand-in covers full-K Python launch glue.

Build both the drop-in and the second lab:
`python prof/lab/build_moe_v9_reuse.py`
Read landed receipts, without querying the judge:
`python prof/lab/read_moe_v9_reuse.py <receipt-or-job> --json <output>`

The two proj lab submission queue keys are8f516be7bb7c5c00 and370b07e5e40e5ad0
(not judge IDs). Humming jobfbf64bd39bde4bf6 also needs its receipt.
`manifest.json` records hashes and local checks; FINDINGS holds decisions and scope.
Candidate sends and commits belong to the owner. This lane sent only lab payloads.
