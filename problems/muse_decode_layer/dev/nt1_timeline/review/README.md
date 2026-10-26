# P3a review: isolated S15 changes

Base: exact S15 `60254883`, original small-M flags95 at M1 and69 at M8/16/32.
`claims.json` pins the actual source/CUDA hashes and each verified code premise.
Each arm changes one mechanism; SHARD2/4/8 have the already-measured PAIR parent, included as a control.
No new arm has an H200 timing receipt yet. P4 receives every valid per-shape point winner with ratio,
SE and seeds; only its held-out six-load R lower confidence bound above1 promotes.

| Item | Confirmed code / implementation | Decision and evidence |
|---|---|---|
| BLOCK | S15's W2 waits on `NT1*nblk[e]`; publications count all blocks of the expert. Index both producer and consumer by `4*e+nb`, targetNT1. 128 done slots (512B), existing control counters move after them. Every W13 store still precedes per-thread fence, CTA barrier, publish atomic and W2 acquire. Only M16/M32 dispatch to this arm. | Built. Empty/skewed/full routes and changing captured inputs pass; new counter slots reset only after every CTA passes its W2 waits. |
| CANCEL | S15 multiplies each W2 result by `a=256*gs2[e]`, then applies post-norm weights and RMSNorm. Cancel a through that normalization: `sign(a)*u/sqrt(mean(u²)+eps/a²)`. W13/SwiGLU unchanged. | Built. Zero scale/zero norm guarded; epsilon/sign metadata retain original FP32 global scales, with double calculation where epsilon's FP32 range fails.160 native-helper extreme cases pass; zero/negative and1e-10/1e10 whole-MoE scale cases pass. No widened NVFP4 values retained. |
| SHARD2 / SHARD4 / SHARD8 | PAIR's norm flush atomically adds into one slot per sorted pair. Replace it by2/4/8 slots indexed by CTA, zero all during W13 tile0, sum them in the finalizer. | Three separate arms. No LOCAL-style per-CTA export or final gather. Parent PAIR stays in the same lab, with direct PAIR ratios as well as S15 ratios. |
| TWO_BASE | The finite sign/exponent assembly identity on S and1.5S exists;8,128 signed finite-format packed-half checks pass. Compare actual16-output BF16x2 old/new decode probes on sm90a. | **Dropped before H200:** integer instructions37→406, registers30→31. `audit_decode.py`, `decode_issue.json`, `two_base_probe.cu`. This closes this assembly recipe; a better packed implementation remains open. |
| PACK5 | S15 stores4,096 scale bytes in each36,864B tile. All100,663,296 actual scale bytes from official layer `generate_weights(5353)` fit32-symbol windows in256-byte packets. | Built. Final format:16 packet bases, raw mask,32B header,160B per compressed packet or256B raw,8B tile offsets,16B alignment. Complete bank including metadata is4.05816% smaller. Timed TMA and exact scale decode; no prepare-time widening. |

PACK5's independent full-bank proof compares all1,610,612,736 weight codes and100,663,296 scale bytes.
`extreme_validation.json` checks the native decoder on17 mixed raw/compressed tiles;
`raw_moe_validation.json` forces raw transfers through9 captured complete-MoE checks at M1/16/32.
The wide/raw layout is larger than the original, as expected; it preserves arbitrary original scale bytes.
The reported4.058% saving is measured on the official generated layer weights, not a claim about a
separate pretrained checkpoint or every possible weight distribution.

`component_validation.json`:90 TASK checks per arm,94 for CANCEL, including zero/empty/all8 routes,
M1/8/9/16/24/32, nonzero expert starts and changing graph inputs. Worst total/row .002391/.002481.
`native_audit.json`: every full flags69 image retains32 HMMA and128 HMUL2; BLOCK/CANCEL/PAIR/shards
use125 registers, PACK5 uses127; zero stack/spills. Two-base decoder is excluded from candidates/lab.
`manifest.json` pins each cell, S15 candidate and exact patch. Candidates regenerate in `runs/screen/`.

All14 normal/forced whole-layer screens pass. Admission runs through `tools/local_eval.py`; `screens.json` records the
exact candidate hashes,13 correctness cases,12 dispatch shapes,26 static checks and eager/captured calls.
Hopper-only attention uses explicit S15 launch stand-ins on sm120; every changed MoE kernel executes
native arithmetic. The real sm90a source still compiles. OOM attempts and the failed packet-audit attempt
remain local logs, not passing receipts. Only successful exact-hash screens can be reused.

`prof/lab/moe_p3a_review.py` packs BLOCK/CANCEL/PAIR/SHARD2/4/8/PACK5 against the exact S15 stage.
It measures16 fresh weight/router seeds at M1/8/16/32, swapped-order read-flushed graph timing,
independent A:A, TASK before/after, direct shard/PAIR comparisons and consumer nodes. Data is a packed
seed-level matrix: event times rounded to1ns, median-of-ratios rounded to1ppm, then losslessly compressed.
`lab_local_validation.json` pins the exact complete payload:64 A:A pairs,all arms/all seeds pass,
6,399-character receipt,52.4s locally. Local times are not H200 evidence.
`receipt.py` round-trips every row; `read.py <receipt.json>` prints ratio,SE,seeds and diagnostic A:A.

Reproduce on GPU0:

```sh
python3 problems/muse_decode_layer/dev/nt1_timeline/review/build.py BLOCK CANCEL PAIR SHARD2 SHARD4 SHARD8 PACK5
python3 problems/muse_decode_layer/dev/nt1_timeline/review/audit_native.py
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_decode_layer/dev/nt1_timeline/review/check_components.py
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_decode_layer/dev/nt1_timeline/review/check_extremes.py
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_decode_layer/dev/nt1_timeline/review/check_raw_moe.py
python3 problems/muse_decode_layer/dev/nt1_timeline/review/run_screens.py
python3 problems/muse_decode_layer/dev/nt1_timeline/review/build_lab.py
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_decode_layer/dev/nt1_timeline/review/run_lab_local.py
```

Sent through `h200_lab.py` on MoE: sender**1191843**,wrapper**`a2344c10505b0e44`**,priority1.2,
import-time budget240s,full9000-character allocation. `queue.json` pins the exact screened send. No sending from this directory bypasses `h200_lab.py`
or `okc_run.py`. The prefill host is excluded for this task.

The first raw-source wrapper exceeded the server1000000-byte cap and was rejected before a job
(sender1113484,`source_limit_rejection.json`). Lossless source packing retains expanded SHAae7b3cf1;
the exact packed payload was rerun on all16 local seeds. The replacement wrapper is975472 bytes.
