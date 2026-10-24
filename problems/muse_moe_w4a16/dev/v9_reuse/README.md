V9 native seed portfolio

`dropin.py` is self-contained: call `moe_reuse_prepare(full_decode_weights)` once,
then `moe_reuse_experts(x_bf16, scores_fp32, local_ids_int32, state)` per step.
IDs are local0..31, remote-1. The component owns one raw NVFP4/E4M3 bank shared
by native and hot-first state. Old E5M3-reencoded banks are incompatible.
The three-input FP32 gate precedes eight cold paired graph measurements; new
choices also require every paired sample to stay below1.05x the incumbent.

`REPORT.md` contains the12-seed local table and the landed V1 H200 table.
M128 hot-first NT2/one stage improves the local mean4.65%; M256 native NT2
improves8.18%. The max<=1.05xmean target is not achieved. Timing means vary
because both touched experts and pairs vary between public model-weight seeds.
Independent-tile reuse, separate-phase queues and W2 split-K lose and stay
outside dispatch. `balanced.py`/`split_w2.py` are diagnostic experiments only.

Build the current drop-in and lab:
`python prof/lab/build_moe_seed_lab.py`
Reproduce the exact-front local seed sweep:
`CUDA_VISIBLE_DEVICES=0 tools/img python prof/lab/moe_seed_sweep.py --hot`
Read a landed native seed receipt without querying the judge:
`python prof/lab/read_moe_seed.py <receipt-or-job> --json <output>`

H200 V1 a029bd95b80a461f: all new kernels lose to V9. V2
3f2b8da0f1d244d7 was accepted already. The new native seed lab is separate;
its generator/fields and its moe_tune routing fixture are explicit in its output.
The new proj submission queue key is56040f879cdeefd2; the send returned no judge ID.
`manifest.json` records current hashes and verification. The owner sends
candidates and commits; this lane sends lab payloads only.
