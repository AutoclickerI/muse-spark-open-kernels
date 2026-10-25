# S3: S2 plus lossless C1g2112 at M256

Candidate: `../../cand_r3_best7_scale_recoded_m256.py`. The frozen source and
payload hashes are in `manifest.json`; regenerate with `python build.py`.
S2 (`b0805447`) is unchanged. Only the two verify32 expert cells change.

H200 job `9e8a6a15f95748db`: 12 public weight/router seeds, three fresh inputs
and eight rotated/reversed cold graph replays. C1g2112 takes 365.2 us mean
(346.4 median) versus raw R1g2112 at 386.1 and masked Humming at 448.2 us. Paired
C:R is 0.946; maximum L2/row errors are 0.00411/0.0048. These are standalone H200
measurements; the scored whole-layer run is `5c7323ea63d944b2`.

The six packing/launch functions and native source are exact copies of the
accepted lab payload (`d30f4a862fa0b882`). Configuration: NT32/K128/four stages,
which=1, grid=2112, encoded=True; GPU route/reset, W13+BF16 SwiGLU, fixed W2 with
sum of squares, then a distributed vector norm/combine. `audit.py` checks the
payload embedded in the actually accepted H200 wrapper, unchanged S2 functions,
both table cells and the rejected-word scan.

Prepare compiles the new module alongside existing native builds and recodes
the packed weight planes/scales once. It retains the recoded M256 bank;
temporary raw planes are released. Scale signs fold into FP4 signs, zero-scale
codes clear, and exponent/mantissa bytes replace E4M3. Both representations
give the same values, including subnormals; dequantization stays in the timed
kernel. No prepared expanded BF16/FP8 bank is retained. S1's NT1 recoded bank,
Humming's independent raw format, projections, attention and glue are unchanged.
The replaced raw CTA engine is removed. M256 admission checks three public
fixtures in forward/reverse fresh-route graph replay against the FP32 oracle;
failure aborts admission.

Reproduce both screens:

```bash
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best7_scale_recoded_m256.py --no-baseline \
  --json problems/muse_decode_layer/dev/s3/normal_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python \
  problems/muse_decode_layer/dev/s3/force.py --cold-native
python problems/muse_decode_layer/dev/s3/audit.py
python problems/muse_decode_layer/dev/s3/write_validation.py
```

`force.py` forces the shipped Hopper table, retaining the native portable math
of the new recoded expert. Hopper-only attention/DG/Humming launches use the
established stand-ins. It requires eager and captured recoded calls for both
M256 full-layer rows and records the changed-input gate errors.

Normal and forced screens each pass 13 tests, 12 scored-shape graphs and
19 static checks, including native SM90a compilation. Prepare takes 9.180 s
normal / 40.954 s forced, including 0.534 / 0.322 s for M256 packing/recode.
Both M256 full-layer rows execute eager and captured recoded math. The public
fresh-route FP32 expert gate reaches maximum L2/row errors 0.003803/0.004261.
The source SHA-256 is
`cadef19a6eced1ae1fa0f3deaffa9ee21e083aa5ba6fe46784707bb8df52a7c7`.

Validation and measured prepare times are in `validation.json`. Local passes do not
establish H200 whole-layer speed or cold H200 preparation time.

The older `assemble_dense.py`/`dense_gate.py.inc` are inactive tooling: the
persistent-chain receipt 87dc944b and FlashInfer receipt 8ffbac2d both lost.
Neither supplies an enabled cell in S3.
