# P4 S15 integration — scheduled, not promoted

Experimental source: `problems/muse_decode_layer/cand_r3_best7_s15.py`, **60254883** (915,762 bytes),
on exact S14 final **f1c77ce6**. The S14 scored send remains unchanged. No S15 final exists before its
paired H200 admission. Experimental asserted handoff: `patch.py`; full hashes are in manifest.json.

## Pieces and evidence being tested

- Attributed NF projection patch **17379ce9**, snapshotted in proj/: O at T1/8/16/32, qkv3 at T1/32,
  full QKV at T32, unpooler at T16/32. Projection-only reproduction is exactly **e3ce8f15**.
  MIT copyright/permission is present in both expanded source and the compressed lab host.
  Stage receipt e3c637d10a584a0f: qkv .707/.817 at T1/32, O .85-.96, unpooler .808/.900 at T16/32;
  these are stage comparisons, not a claimed S15 layer gain. Fused pool/router remains S14.
- Last-CTA B1 sliding patch **96e2c5b3**, rebased from exact S11 with all old/context lines asserted.
  Only decode1 sliding changes. S13's direct-partials decode1 FULL remains; its sliding prepare row is
  removed when last-CTA is enabled. Stage receipt7e93cc81af17489a: .98017 vs direct-partials.
- Register-fold full verify patch **ca730e61**, stage4 at B16/32. Source/rows/module name are replaced
  against asserted S11 values, then its prepare admission is mandatory. Stage receipt9d9a01e1f6db4d79:
  .97726/.94813. Official changed-full host8e091fcb is queued separately; this complete candidate also
  receives official changed-full correctness in the integration run.
- Metadata-only **f39bc1f3** is excluded. Its6cb428bc shared-JIT confirmation remains scheduled, without
  an accepted job/result at integration time. Do not add it from the provisional+.3% estimate alone.

## Local admission

All heavy GPU work ran serially on GPU0. Combined normal/forced each pass **13 TASK tests /12 graph
shapes /25 static checks**. Every isolated piece independently passes both modes: projection24 static,
last25,regmax24. Full recursive prepare inventory and all9 NF layouts pass; see PRECISION.md.
Normal/forced call records confirm all9 selected projection rows, last-CTA sliding, direct-partials full,
and both full-verify shapes. Source/proof/dependency hashes are pinned in admission.json.

Full wrapper **49ce378b**,634,595 bytes, passes normal/forced local_eval13 TASK tests plus its bootstrap
static checks, AND every internal12-shape x16-seed output/capture/compile gate. The expanded source's
native25-check audit remains separate and complete. Each arm runs every shape, including controls.
Seven off arms: proj,last,regmax,qkv,O,unpooler,S14 back; A and A2 provide A:A.32 cold rounds at T<=8,
8 elsewhere. Seeds3121907+7919*i are fresh vs S14. The full1GiB cold-read flush is retained locally.
Each arm shares35 unchanged Triton objects/device functions; source dependency closures must match.
Changed VQ8 source cannot inherit the new host module in its old-source off arm. Exact S14 projection
plans/GEMV choices/MoE banks/counters remain unchanged. Compact RESULT records all shape/load ratios
and A:A envelopes in under8,500 bytes, so they survive the judge tail.

Local-only unsupported last-CTA resource query and initial bootstrap F821 failures were fixed before
queueing. Failure metadata remains under trials/. No local failure was sent and no check was suppressed.
The new native attention CUDA compiled for sm90a; actual native H200 execution is still pending.

## Reset queue / coordination

`queue.json` records exactly one combined **check**, priority **-1**, held until
**2026-10-26 00:00 UTC /09:00 KST** on the decode host. Detached sender PID3526702;
log `runs/detached/49ce378bb4bbb287.log`. No accepted H200 job ID yet.10 scored slots are preserved
for candidates. Scheduled wrapper SHA256 **49ce378bb4bbb287731b16a0ab6ac99f5b165c65697d7d98b69438f5c0aa1260**.

P10b's redundant attention-host selected-only layer batch **c4c247dd** was cancelled before acceptance
through okc_run.supersede, after verifying the live combined queue, exact old source, same account/cwd
and waiting entry. No accepted job or foreign-project process was touched. Its original pool/qkv16 probe
source remains in public_nf/pool_q16_lab.py; requeue those probes separately when useful. Coordination
and the former queue entry are preserved in coordination.json and public_nf/README.md.

The helper now accepts --base-file and --check so this new candidate is officially validated as host
without spending scored quota. The committed helper reproduces this exact wrapper; unrelated import-phase
helper edits from another lane were left outside P4's commit.

## Reproduce / resume

```
python3 problems/muse_decode_layer/dev/s15/build.py
python3 problems/muse_decode_layer/dev/s15/ab_build.py
python3 tools/h200_lab.py build problems/muse_decode_layer/dev/s15/s15ab_payload.py --host muse_decode_layer --base-file problems/muse_decode_layer/dev/s15/s15ab_host.py -o problems/muse_decode_layer/lab/lab_s15ab_payload.py
```

The scheduled source is already pinned; do not queue it again. After acceptance, find the job by wrapper
SHA in runs/pending or ledger, collect through tools/okc_run.py, then run `dev/s15/report.py <job>`.
Promotion requires every retained piece/load to win beyond the full A:A envelope, unchanged shapes and
loads within A:A, and S14-back wins. Keep raw failures; do not promote by the outer host score.
Next work waits on reset/native paired evidence. Select or drop pieces, screen the resulting source, and
freeze/report its exact path+SHA only when its promotion gate passes.
