# M128 instruction evidence

H200 **dc3fea0a20a349bf** is complete: NT1rec290.07, NT1lop280.70,
D16word294.87, D32word302.08, **D32g272.95 us**,36/36 fixtures over12 seeds.
D32g mean paired ratio0.9465; maxL2/row0.00413/0.00469. Seed0/fixture0 cold
route/W13+act/W2+SS/finish graph medians7.8/174.2/81.0/12.4 us.
NT1 W13 per-CTA min/median/max113.2/130.7/161.8 us are instrumented intervals,
not a standalone phase wall time. The phase-mix lab measures that comparison.
A:A abs mean7.956us includes a1353.216us raw outlier; fixture medians temper it.
The exact D32g cell is handed to decode and source-pinned S4; see
`../phase_mix/d32g_handoff.json` and `h200_result.json`.

The combine-first **640d67e7e56d42e5** also completed: best expert-wide RS361.16
vs NT1rec290.83 us, shared-A678.86–785.54. Logical one-read alone does not
make these implementations faster. All numeric gates pass; no one-read cell
is promoted. The paired experiment `../phase_mix/` **770809cf159f41e2** confirms H_H
234.44us vsD32g272.70 and NT1rec289.62: NT1<=16 and WGMMA32>16 in both
phases. Its fixed cell is handed to S5; threshold/N64 tuning runs on
proj as job `4c469bf9` (`../mix_sweep/`).
Static issue estimates below do not establish the binding hardware limit.

Measured here: sm90a SASS from nvcc -O3, nvdisasm -c -g and cuobjdump --dump-sass, with source hashes and exact PC ranges in audit.json. Native entries have no spills/C7513; 16 K64 register-bank paths remain safe.

Instructions per 16 weights per lane (the fused rows include activation gathering/copies in their K loops):

| Path | Unpack | Scale (incl BF16x2 multiply) | MMA | Load/copy | Address | Control | Total | Canonical-byte issue estimate TB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| norig_K512 | 17 | 10 | 2 | 0.625 | 0.6875 | 0.4375 | 30.75 | 9.99 |
| nlop_K512 | 16 | 10 | 2 | 0.625 | 0.625 | 0.5 | 29.75 | 10.33 |
| ibase_W13_N32 | 17 | 16 | 2 | 6.5 | 37 | 6.25 | 84.75 | 3.62 |
| iword_W13_N32 | 16 | 12 | 2 | 7.5 | 37.5 | 6.25 | 81.25 | 3.78 |
| idist_W13_N32 | 16 | 12 | 2 | 5.25 | 25 | 4.75 | 65 | 4.73 |
| idist_W13_N16 | 16 | 12 | 2 | 4.75 | 19.75 | 4.5 | 59 | 5.21 |

NT1 is a straight-line unrolled K512 reader: 492 ->476 instructions for 256 weights/lane, a 3.25% cut. The explicit LOP3(A,mask,B,0xe2) replaces the compiler's two-instruction blend. The existing tiny-BF16 bit-plane representation and BF16x2 multiply are already the magic-number conversion; scalar PRMT lookup is not substituted.

The repaired wide producer's adaptive N32 W13 loop falls 84.75 ->81.25 instructions/16 weights (4.13%). Its address work is 37/84.75=43.7% of the baseline static loop. The smaller fixed N32/N16 loops use 65/59 instructions/16 weights. Fused loop counts include uniform-pipeline operations and all static natural-loop bodies: conditional drain/refill paths and predicates mean they are not an exact dynamic retired-instruction count. The estimate is a scheduling model, not measured throughput or a proof of one hardware bottleneck.

The model uses the supplied 4 warp instructions= 128 lane-ops/SM/cycle and 15 B/SM/cycle at 4 TB/s. For n instructions/16 weights, canonical NVFP4 bytes/SM/cycle=128*9/n and the issue estimate=307.2/n TB/s. Word banks physically hold 10 bytes/block, so their encoded-byte estimate is 341.333/n; their canonical-byte bandwidth ceiling at 4 TB/s is 3.6 TB/s. Functional-unit throughput, stalls, active masks, uncounted setup/epilogues and synchronization can only be resolved by more evidence.

Local public routing fixtures touch 16.139 experts on average (456.917 MB unique NVFP4). Weight-tile decode rereads per touched expert at NT8/16/32/128 are 2.176,1.478,1.157,1.000. NT1 issues 972.816 MB of logical tile reads per call. Without L2 tile reuse this alone costs 243.2 us at 4 TB/s; the ideal aggregate-issue work costs 97.4 us. These are computed on this lab's fixtures, not a reconstruction of 58ae2e2c's different full-layer draws. L2 can save traffic but not repeated decode instructions.

This evidence does not establish aggregate issue as NT1's sole binding limit at M128. The unrolled-reader estimate is 9.99 TB/s before route rereads, above the HBM target; epilogues, stalls, data reuse and individual execution pipelines remain possible limits. It does expose excessive address/transport work in the wide fused producer and motivates the fixed smaller tiles.

The u16 block-scale metadata is exactly 0x7000+(u8_recode<<4), stored as int16. It preserves the original E4M3 cardinality and is reversible; E2M1 codes stay packed. Sign/zero folding is unchanged. No expert-weight matrix is dequantized in prepare. Four PRMTs replace four PRMT+LEA pairs per 16 values. It costs one extra metadata byte per 16 weights (+11.1% bank bytes). Q copies widen 8 -> 16 bytes and scale copies 4 -> 8 bytes; the paired H200 lab decides whether this tradeoff pays.

Local phase medians below are SM120 immediate-MMA correctness diagnostics. They cannot rank Hopper or supply the requested H200 phase claim. Each fused phase is a separate cold CUDA graph replay with predecessor phases refreshed before the flush. The NT1 diagnostic reports per-CTA globaltimer intervals; these overlap across CTAs and must not be summed as maxima. Its W2 interval includes inter-expert dependency waits.

- PH F128rec route/W13_act/W2_final=7.2,435.3,1206.3
- PH F128word route/W13_act/W2_final=6.1,423.9,1252.4
- PH D16word route/W13_act/W2_final/finish=6.1,416.8,220.2,14.3
- PH D32word route/W13_act/W2_final/finish=6.1,439.3,220.0,14.3
- PH D32g route/W13_act/W2_final/finish=6.1,418.8,215.1,14.3
- NTPH globaltimer us perCTA min/median/max route,W13_act,W2,barrier,combine grid=188 7.1/7.6/9.8,185.3/216.9/226.5,96.1/114.4/146.3,1.1/12.4/29.6,0.5/2.0/4.0

The new lab compares NT1rec, NT1lop, F128rec, F128word and D16/D32 word-bank producers (including grid 1056), 12 weight/router seeds x3 identical real-route inputs, six cold 128 MiB reversed/rotated replay orders and A:A noise. Independent finite/FP32 gates run before and after. D16/D32 reuse the free lane's distributed W2 finalizer. No runtime workload is skipped.

Validation: 180 producer graph checks, 72 NT1 graph checks, 42 additional signed/zero/subnormal and route-transition checks, 4064 exact scale/code values; wrapper 5 tests/6 shapes/6 static checks. Local payload 26.4 s within 78 s.

On resume, read dc3fea0a20a349bf once. Require complete seed means and every gate, compare gains to A:A noise, then enable only a winning M128 cell and pass the decode lane its path+SHA. dropin_nt1_lop.py is locally gated but disabled; its exact handoff is in dropin_manifest.json.
