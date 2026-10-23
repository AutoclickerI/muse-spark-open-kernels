# Projection handoff: decode first

The fitted overall score is max(decode, component composite); standalone proj currently
cannot move our rank. Its kernels now serve the complete decode layer and serving.
This round is solo, GPU0 only. The orchestrator sends/commits; this thread did neither.

## Ready next

**`problems/muse_decode_layer/cand_r3_proj.py`**

Frozen SHA256: **`235434b35d1407cdb61ff79f291b9a962fa4a56369f60277fb666863f5b1174e`**

Parent: `cand_r3_v9_autotune_q1.py`, SHA256
`099ee216693f890b1bc6b737de5d3441e5f210baa2cca0c8fbb0b4fd65ecbc6e`.
Only this owned decode source changed. Parent and the decode lane's other sources stay intact.
The final file keeps V9 as MoE control following new complete-layer evidence, rather than
rerunning the inherited synthetic MoE ranking that mispicked the verify32 path.

| Validation | Result / evidence |
|---|---|
| Official local screen, **sm_120** | **13/13 tests, 12/12 fresh-input graphs**, prepare **29.05 s**; exact frozen SHA in `runs/screen/decode_proj/muse_decode_layer.cand_r3_proj.json` |
| Native projection replay | **2,742** configurations, three fresh inputs each, including outliers and a one-column probe; `dev/decode_proj_validation.log` |
| Forced Hopper glue | **730** cases across paired BF16/router projections, all 27 FA3 settings ×6 shapes ×tensor/tuple returns, and q_len1/8 split-qkv tuple consumers; same log |
| Reject/budget injection | **16** injected bad FA3/projection trials rejected; expired deadline retains proven controls; same log |
| Hopper build | Complete projection CUDA module builds **sm_90a**; `dev/decode_proj_sm90a.log`. Stand-ins do not verify WGMMA execution |
| Rebuilt standalone tuner safety | Coarse launch/numeric, finalist and layout failures rejected on all 30 shapes; expired deadline skips all trials; `dev/tuner_safety_validation.log` |

The projection registry covers 24 scored `(T,name)` keys: T=1/8/16/32/128/256;
qkv_gate, o, pool_router, unpooler_slice at T<=8 and unpooler otherwise.
It includes measured skinny/GEMV schedules, single-pass hi+lo in shared FP32 accumulators,
a single mixed pooler/router kernel, strided pooler Lt + fused router (no packing/add kernel),
TMA/WGMMA direct/last-CTA/DSMEM/Stream-K and all returned checked Lt TN/NN algorithms.
Projection trials time the next consumer. Every result keeps the layer's `(P,S)` interface.
Trials validate two fresh inputs against cuBLAS within 1%; FA3 uses the proven default within
1%; PDL checks the proven enabled path. Bad variants are logged and removed, not raised.
The shared **240 s** deadline stops trials early; untried shapes retain proven controls.

New H200 decode evidence: pure V9 `0e94268cf5654012` **1.427904x** (CI1.406485–1.449322),
versus grouped dispatch `2ad75c19cb514f5c` **1.265668x**, with a verify32 collapse.
That evidence motivated keeping V9 for all token counts and skipping its unused V7 bank.
The final file's H200 correctness, projection choices, complete-layer speedup and cold
prepare time remain unmeasured. Local timing is admission evidence, not an H200 prediction.

Reproduce:

```sh
CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_decode_layer/cand_r3_proj.py --json runs/screen/decode_proj/muse_decode_layer.cand_r3_proj.json
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/validate_decode_proj.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/validate_decode_proj.py --build-only 9.0a
```

Builder: `dev/build_decode_proj.py`; embedded templates: `dev/decode_proj_tuner.py`,
`dev/decode_proj_host.cuh`. The builder refuses to replace an existing candidate.
`portfolio_all_tail.py`, `portfolio_v2_tail.py` and max/native builder inheritance now
carry rejection and 240 s budget handling into generated copies. No queued source changed.
`validate_tuner_safety.py` builds a fixture in a temporary directory and injects failures;
it is development evidence, not a candidate to send.

## Already sent projection portfolio

Only **`cand_portfolio_all.py`** was sent by the orchestrator. Its owner-patched source is
SHA256 **`0caf732223b826313b80e5f73bfa9d608c7c4290eaa3ebc84233df3ac23c1453`**,
waiting pid **214968** at priority2 when inspected. The former 65ea3171 local admission
belongs to the earlier source, not this patched hash. Read its receipt when it lands.
Do not resend or edit it. Native/max/h200 standalone portfolios are subsets for this
round; they are not additional recommended sends. The projection-board 1.26x target
is superseded by complete decode improvement for G3.

## Existing queued diagnostics (do not resend)

Queue snapshot from `runs/waiting/`; the orchestrator owns subsequent state changes.
Their distinct per-shape results can feed a later decode/serving revision.

| Source | Frozen SHA256 | Queue state |
|---|---|---|
| cand_dsv3_select.py | `0453c6ae3dd3b05a46846f556da94290a0b20b31a0c158e354dae72d1523b4c2` | waiting `128977` |
| cand_dsv3.py | `8d9d827cd15b00b22a898d851f1a49e04450fb15cbad494a6d1cf19ca011a624` | waiting `128979` |
| cand_gemv3.py | `16ba69bb100b069ca53d03096057f20261561feeb689a2139a7215f39178c9ec` | waiting `128983` |
| cand_m16.py | `48c5397b78df800e9cc843bdd0659c5475bf745ca65c0184c81d8fea5bff017e` | waiting `128986` |
| cand_skinny.py | `89bb6387a6d43ebf53cae10faee99464b30b6375a3daf45d56b99b1b64438ee8` | waiting `128991` |
| cand_tile.py | `4f8b1aa3aa9974b203b85ba68189f3c57ccf2c890f31fb9a789f6b486aa20cb7` | waiting `128993` |
| cand_tc.py | `c0553a0bb0b060f3f2b74b8d42b938e7ad154a748250d1c43ad9f865e7504cf3` | waiting `128997` |
| cand_tc_k256_qkv.py | `0036aeba8960ce0ea0585968ae726a4464339f5db4eea16f3233bde7a8093120` | waiting `129011` |
| cand_gemv_static.py | `f50425383a9c75e206fbf5d13202ac7ff6c7883087c6900acb9a6a3ab7e7a76a` | waiting `129014` |
| cand_tuned2.py | `40648053bc6269f2010db7f38d2b92e458498faa94334b31482613c4a525883b` | waiting `129016` |
| cand_tc_k256.py | `accd2952ae08baf952e9cced1c2418e94094a1af3ccbb919479d8077f2ae7779` | waiting `129028` |
| cand_wg2.py | `b02135be25cb7918d76252387df63b2bb30e0070a1d2d7006727bf535196fcd7` | waiting `129031` |
| cand_wg.py | `c224d1a5bae0fb7a0164d06e82d185c2271e6c4778b5a70c7d34d6562f6fbe8e` | waiting `129034` |
| cand_reduce_hier.py | `051b1eeb33f0bd1cc38a19ff8273eaad2a9254fba92529e8be869b464ebbef33` | waiting `129153` |
| cand_reduce.py | `280822ad5f91aae8e20e9440032f3f3ce22043119179375c4087059ee0c887f6` | waiting `129168` |
| cand_lt_layout.py | `6c65aff44f84301c8eb9b772a1a2ac2ab3cece9429e44a2c6d0adb207d2c0381` | waiting `129180` |
| cand_reduce_fast.py | `b28e00958daef31379f8e3df4698e43be3ae05487c0e09c7d36204248132413e` | waiting `129185` |
| cand_reduce_packed.py | `d19dcd2bc9c0d036297bc70474318aac6e465d83267d88dff9add5ca78d5211e` | waiting `129202` |

Completed fused-select `f3786341611f4f1f` covers the CUDA bodies/configurations in queued
`cand_fused.py` and `cand_fused_lean.py`; they add no new H200 evidence. This thread did
not cancel them. The queued wg/wg2 files never set `wg_ok`; their inactive branches
are explicitly exercised in the broad portfolio instead.

All **14** completed clean projection receipts through `f3786341611f4f1f` were read;
no later completed projection receipt was present at handoff. Standalone board/source
remains **1.181998x**, SHA256 `10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9`.
`docs/FINDINGS.md` owns the 30-shape SGLang/best table and H200 mechanism evidence.
Next: run the frozen decode candidate, read `PROJ_LAYER`/`PROJ_TUNE` selections and the
complete-layer profiles, then combine only H200-winning choices with the decode lane's
attention work. Serving projection ports are a separate active lane; no serving file changed here.
