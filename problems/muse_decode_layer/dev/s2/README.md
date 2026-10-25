# S2: S1 plus M256 CTA1g2112

Candidate: `../../cand_r3_best7_scale_w2fix.py`. The exact source hash and frozen
H200 payload provenance are in `manifest.json`; regenerate with `python build.py`.
S1 (`42f543a0`) remains untouched. Only the two verify32 expert cells change.

The raw NVFP4/E4M3 producer is copied from payload `857ee3b1763f85af`, receipt
`97be11d4631b44d6`: 12-seed mean 385.2 us versus masked Humming 447.6 us.
Configuration: NT32/K128/four stages, which=1, grid=2112, mode=1; route/reset,
W13+BF16 SwiGLU, W2+Y/sum-of-squares, distributed token CTA finalizer.
Packing, CUDA source, launch functions and schedule are the measured definitions.
The raw bank is independent of S1's recoded NT1 bank and Humming's own format.

Prepare compiles the pointer-only module with the existing native builds,
then builds packed raw planes and E4M3 sidecars. A mandatory three-fixture
FP32 oracle gate checks forward and reverse fresh-route graph replay. A failed
S2 gate aborts admission; it cannot silently select Humming. The owner can use
S1 if S2 fails a screen. No dequantized weight bank is retained.

`audit.py` verifies exact kernel/runtime provenance, unchanged S1 compute,
the two table edits and the submit-time rejected-word scan. `force.py` uses
established portable stand-ins for Hopper attention/DG/Humming, while the new
expert runs its real portable CUDA branch and records full-layer calls.

Reproduce normal and forced screens inside `tools/img`:

```bash
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best7_scale_w2fix.py --no-baseline \
  --json problems/muse_decode_layer/dev/s2/normal_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python \
  problems/muse_decode_layer/dev/s2/force.py --cold-native
```

H200 timings above are the standalone paired receipt; the scored whole-layer run
is `4e212ec25109434d`.

## Admission result

Source SHA-256: `b080544708544fd67ee23b82aca111bf0b9bb82c30a5eb86315044cd9bf92337`.
Normal and forced: **13 tests, 12 fresh graph shapes, 19 static checks each**.
Prepare14.075/39.603 s; additional packed-bank preparation0.328/0.322 s.
An initial run included cold Hopper attention compilation and prepared in82.305 s
(repack0.452 s); its post-screen counter assertion was wrong because Python calls
do not count CUDA graph replays. The corrected coverage check explicitly records
native captures for both sliding/full M256 rows; the full screen was repeated.
Three expert graphs capture (one admission, two full-layer), with changed-input
replay checked by the oracle and official layer tests. Ported expert gate maximum
L2/row: **0.003803/0.004261**. See `validation.json` for source-pinned summaries.
