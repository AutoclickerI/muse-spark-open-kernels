# S6: mixed_parallel M256 and M1 O/config6

Candidate: `../../cand_r3_best7_s6.py`; frozen base S5 `f7064fc0` is unchanged.
`build.py` extracts the payload actually accepted as H200 job
`8483c0c6783a4df1`, SHA-256 `cb20858c5de9bfaca16ee289529cc2f46554ed575e32a7ce16dcccd31b1cd179`.
`manifest.json` pins the exact source and configuration; `audit.py` verifies
all copied native source, packing, launch and stream/event definitions.

The twelve-seed H200 result is344.5 us mean/330.8 median for mixed_parallel,
versus recoded C1g2112366.2 us and raw CTA1g2112388.0. Each fixed variant sees
three fresh inputs and eight rotated/reversed cold128MiB graph timings per
weight/router seed. Max L2/row errors0.00412/0.0046; all twelve seeds pass.
These are standalone measurements, not the combined S6 score.

The selected policy is6, grid2112, with recoded W13 and W2. W13 uses vector
Q/scale loads and one-LOP3 decoding, N16 for experts with<=16 rows and N32
above16. The two W13 kernels run in parallel on the measured parent/side-stream
schedule, with events forked after routing and joined before W2. W2 and the
distributed finalizer keep their measured recoded source. All routing and
synchronization stay inside the timed call; prepared streams/events contain no
input-derived state. Native scratch owners and current-stream lookup are exact.
The replaced S3 M256 engine and bank are removed. Small-M LOP3, M128 H_H,
attention, glue and other-shape policy stay S5's.

The optional projection is a clean role swap: only O at the two M1 decode rows
selects the supplied `Best7ProjectionCell`, config6, BF16 output. Its full source
SHA-256 is `c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe`.
It uses the existing ('stream',6,1) projection interface, with its own mandatory
three-input FP32 graph gate. Other roles and shapes retain S5's dispatch.
H20095ab383d measured15.39 ->15.04 us with consumer, paired ratio0.9783;
this is close to timing noise. Sliding/whole-layer H200 benefit remains unmeasured.

The forced screen additionally compares this exact cumulative source against
its ordinary O projection on3 changed full-layer inputs in each M1 layer type,
requiring exact live routing IDs and hidden/stream error<=0.003, row<=0.015.
It requires eager and captured calls to the supplied O cell in both rows, native
mixed_parallel at both M256 rows, H_H at M128 and LOP3 on the other8 rows.
Hopper-only attention/DG/Humming calls use the established stand-ins; all shipped
MoE and O-cell arithmetic runs natively on the local GPU. SM90a compilation is
checked separately for every native source.

Reproduce:

```bash
python problems/muse_decode_layer/dev/s6/build.py
python problems/muse_decode_layer/dev/s6/audit.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best7_s6.py --no-baseline \
  --json problems/muse_decode_layer/dev/s6/normal_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python \
  problems/muse_decode_layer/dev/s6/force.py --cold-native
python problems/muse_decode_layer/dev/s6/write_validation.py
```

Both screens pass13 tests/12 scored-shape graphs/22 static checks. Prepare
takes10.203 s normal /50.812 s forced, with0.538/0.326 s for the M256 bank.
The forced time includes the extra six full-layer O comparisons, whose maximum
hidden/stream relative errors are0.001905/0.000952 with exact routing IDs.
The O FP32 graph gate is0.001676; the mixed_parallel FP32 L2/row gate is
0.003803/0.004261. Both changed rows execute eager and captured native math.

Final source-pinned checks and prepare times are in `validation.json`.
Whole-layer H200 speed and cold H200 prepare remain unmeasured.

S6 runs as scored job `fecc29af233d45b3` (send log `runs/detached/4c98f12fc7fab973.log`);
source SHA-256 `4c98f12fc7fab9730b90d3bae2f7f014ce85bae35439fb688282d98bd167c59f`.
The lever parser (`dev/levers/build.py`) attributes the new O kernel separately and groups
all H_H/parallel MoE kernels; overlapping kernel durations remain optimistic headroom.
