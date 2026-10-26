# S19 timeline for lever re-rank

Measured H200 receipt **9c16c850f7b941a5**: [per-stage critical/roof/excess tables](result/H200.md),
[all round timings](result/rounds.csv.xz), [source and accounting validation](result/validation.json),
[ranking, priors and mechanism receipts](result/ranking.json),
[canonical judge receipt](../../../../runs/muse_prefill_layer/20261026T114928Z_check_9c16c850f7b941a5.json).
The lab completed all72 TASK checks and108 profile samples; the outer lab-only host deliberately stops
at prepare after preserving evidence. This diagnostic makes no scored S19 or new kernel promotion claim.

The observer reuses `../s15_timeline`'s CUPTI activity/DAG, completion-chain algorithm, captured-role
mapping, logical-work model and compact trace codec. It executes the checksum-pinned expanded S19
`62ea5bf4` (shipment `8b415553`) without device changes, prepares fresh public synthetic weights/router,
and profiles all twelve scored shapes in one import-time lab. No result feeds a scored timed kernel.

`manifest.json` pins source, observer, independent TASK checker, payload, wrapper and prospective seeds.
Three fresh input seeds per shape, three cold CUPTI rounds per seed; all 108 profile rounds and 288
unprofiled control replays survive at integer-nanosecond precision. Both untouched and mapped captures
pass public TASK on every fixture (72 checks). Each shape's full normalized kernel/resource/stream/DAG
fingerprint must match. Activity includes memory operations, with no added in-graph events or kernels.

Logical completion-path gaps covered by GPU work go to that owner. Mixed expert phases get a separate
MoE-overlap row and one shared MoE resource envelope; genuine GPU idle is separate. Noncritical expert
branches can still consume bandwidth. A phase's critical increment is not its residence or exclusive
instruction-pipe cost. Small-M fused W13/W2/finish remains one row; W2's inline finalizer is not invented as a
separate interval. Attention includes query prep where it is fused or launched inside the attention stage.

Roof: max(logical read+write bytes / 4.8 TB/s, all tensor FLOPs / 989 TFLOP/s), with no measured-stream
intercept. The requested 1.3x-roof excess and the ideal-roof ceiling are separate scenarios. Original
BF16 and selected lossless NF/CD layouts are counted at their actual stored bytes; S19's raw P10 pool
bank overrides the older coded-bank model. Scratch/repeated reads, scalar reductions and softmax issue
can explain gaps above these resource roofs. These are not hardware DRAM counters or attainable-gain
promises. Lossless coding can lower the current-format roof; the exact12 model is explicitly optimistic.

Reproduce:

```bash
python3 problems/muse_decode_layer/dev/s19_timeline/build_lab.py
python3 tools/local_eval.py problems/muse_decode_layer/dev/s19_timeline/generated/s19_timeline.py --payload --tier static
CUDA_VISIBLE_DEVICES=1 tools/img python problems/muse_decode_layer/dev/s19_timeline/screen.py
python3 problems/muse_decode_layer/dev/s19_timeline/verify_archive.py
python3 tools/h200_lab.py send problems/muse_decode_layer/dev/s19_timeline/generated/s19_timeline.py --host muse_prefill_layer --lab-only --import-budget 780 --lane P4 --decision "S19 timeline for lever re-rank" --note "Pinned S19, all 12 shapes, per-round times and TASK on both arms"
python3 problems/muse_decode_layer/dev/s19_timeline/report.py RECEIPT.json --out problems/muse_decode_layer/dev/s19_timeline/result
```

Local forced-Hopper launch stand-ins test the actual Python glue on sm120. The screen's counter wrappers
are removed before namespace cloning because they hold original functions in closures; leaving them in
makes the cloned RT_ARM dispatch read another namespace's value. This is local-only; no S19 device code
or dispatch is changed on H200. `local_validation.json` verifies exact reconstruction of every stored
round, representative timestamp/DAG/resource and logical work tensor model.

Codec v2 drops duplicate median-stage vectors and median wall sample lists: the per-round owner vectors
already retain all timing information requested, and representative full traces allow chain recomputation.
All round values live in the same varint/LZMA archive, so the wrapper's complete packet fits its receipt
reservation. No input seeds, rounds or checks are removed to meet that limit.
