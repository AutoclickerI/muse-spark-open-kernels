# Projection next checks

P7 serves G1/G2: decode M1-32 and glue, plus verify dense inside the decode layer.

| Order | Artifact | State |
|---|---|---|
| 1 | `dev/l2_overlap_lab.py` | Queued on prefill cb8939d0, priority0.5; send key **9367398c624b7796**. Immediate detached log: waiting for a slot; no static rejection |
| 2 | `dev/l2_overlap/read_receipt.py <job>` | Read section `l2_overlap_lab` once it lands; paired whole-layer means/three draws and A/A noise decide the best7 projection cell `{'l2_prefetch': id}`. Keep disabled until H200 evidence |
| 3 | `dev/l2_overlap/PROFILE_MAP.md` | Best6 role/floor map. Saved profiles group pool+slice at M1 and two pairs at M16; activity profiles in the new lab resolve them. Own-weight prefetch is explicitly False at these custom shapes |
| 4 | `dev/streaming_gemv_lab.py` | Landed **feb7b287e3464e4a**: M1 matches same-byte read, all M>=8 lose. Old queued d962/b104 labels are superseded |
| 5 | `dev/streaming_glue_lab.py` + verify dense | Landed **5535d2f171b34d72**: fused glue loses2.3–21x; verify WGMMA loses1.28–2.18x. None promoted; old send keys be5d/5526 superseded |

New prefetch payload SHA256:
`e62b6b087adeb0c5744c79f3f0597039e799f316908dc82951c9d9ee0e3fda55`.
Judge wrapper SHA256:
`9367398c624b779601edfaac312f3ae9337d34653f5e50190ad93e25687ed515`.

`dev/l2_overlap/validation.json`: all78 coarse recipe/cases,12 final pairs,
6 native activity profiles and5 host tests; no rejection/incomplete stage;
12.0s cached local payload in80s. Raw/router FP32 gates remain1e-4;
whole mixed-BF16 layer bound0.3% with exact IDs. A no-prefetch twin exposed
1.04e-4 residual drift at M16 full before prefetch, so the router-only bound
was inappropriate for the mixed-BF16 final stream. Independent stage gates
are retained. Native sm90a:128-bit LDG, UBLKPF.L2, zero spills,44/18 registers.
Best7's switch is disabled pending H200; frozen source08bbc928 passes13 official
tests,12 graph shapes and16 static checks with6 real prefetch instances and
24 streaming instances. `dev/l2_overlap/best7_admission.json` pins this evidence;
the decode owner's later default-table edits require its cumulative screen.

Earlier fused implementation evidence (not a promoted kernel):

Fused payload SHA256:
`8ec41783b2300e77a48e456124aa7502a89731c0dcbad3bdfd643d45b7feb188`.
Generated judge source `problems/muse_prefill_layer/lab/lab_streaming_glue_lab.py`:
`552621abdd6a86a3994815039ef792da22e87320704aec6ebcecdde092b78c68`.

The streaming library now completes o/post-attention RMS/residual/pre-FFN,
pool-router/pre-expert RMS/routing, and unpool/post-FFN RMS/residual/next-pre-attn
in each projection's launch, including split-K reduction. Config+200 distributes
FP32 quadratic moments across the projection CTAs. The alternate pre-FFN
placement normalizes the o FP32 stream in the pool/router prologue and rounds
to BF16 before MMA. Qkv, M1/8/16/32/128/256, two consumers and both moment modes
share one lab. The independent raw FP32 gate is 1e-4, including router hi/lo,
outliers and single-column probes. Route/local IDs are exact against its FP32
prefix; all captured counters reset. Cancellation edges use an FP32 forward
error bound relative to both residual addends, plus the raw and final-output
gates; a residual-relative comparison alone magnifies normal FMA rounding.

Local source-pinned evidence in `dev/streaming_glue/`:

- `validation.json`: **6,105** configuration/edge checks, **510** captured
  graphs with fresh replays; all direct and moment paths, all consumers and
  split counts, M/K/N tails, actual shapes, score ties and residual cancellation.
- `compile_sm90a.json`: **33 kernels**, no entry or device-continuation spills,
  max240 registers, native 128-bit/async loads and BF16 MMA. Smaller ring stage
  count locally on sm120; actual H200 stages compile unchanged.
- `host_admission.json`: **5/5** official prefill host tests, **30/30** stage
  keys, **12/12** dependent chain pairs, no rejection, **34.4 s** cached local
  prepare (payload28.8s), **5,889** lab characters. Each finalist has20 AB/BA
  cold256MiB pairs against cuBLAS+glue and the same GEMV with separate glue.

The fused H200 receipt above resolves against promotion. Cross-stage prefetch
remains unverified on H200.
The queued sources are immutable evidence; rerun the generators only for a
new measured variant. Original standalone admission is in `dev/streaming/`.
